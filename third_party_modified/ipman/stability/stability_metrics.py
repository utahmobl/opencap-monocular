import torch
import sys
import numpy as np
import trimesh
from scipy.optimize import linprog
from part_volumes import PartVolume
from third_party_modified.ipman.vis_utils.mesh_utils import GMoF_unscaled, HDfier
import pickle as pkl
import time

SMPL_PART_BOUNDS = (
    "data/essentials/yogi_segments/smpl/part_meshes_ply/smpl_segments_bounds.pkl"
)
FID_TO_PART = "data/essentials/yogi_segments/smpl/part_meshes_ply/fid_to_part.pkl"
PART_VID_FID = (
    "data/essentials/yogi_segments/smpl/part_meshes_ply/smpl_part_vid_fid.pkl"
)
HD_SMPL_MAP = "data/essentials/hd_model/smpl/smpl_neutral_hd_sample_from_mesh_out.pkl"


def stability_error(vertices, faces, contact_thresh, cop_w=10, cop_k=100):
    with torch.no_grad():
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = compute_per_part_volume(vertices, faces)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = HDfier().hdfy_mesh(vertices, vertices.device)

        # create face_id to part label map
        with open(PART_VID_FID, "rb") as f:
            part_vid_fid = pkl.load(f)
        face_id_to_part_label = face_id_to_part_mapping(part_vid_fid, faces)
        with open(HD_SMPL_MAP, "rb") as f:
            faces_vert_is_sampled_from = pkl.load(f)["faces_vert_is_sampled_from"]
        all_fids = torch.arange(len(faces), dtype=torch.long)
        hd_vert_on_fid = all_fids[faces_vert_is_sampled_from]
        hd_vert_label = [face_id_to_part_label[fid.item()] for fid in hd_vert_on_fid]

        # calculate volume per vert
        volume_per_vert_hd = torch.FloatTensor(
            [per_part_volume[part] for part in hd_vert_label]
        ).to(vertices.device)
        if volume_per_vert_hd.dim() == 1:
            volume_per_vert_hd = volume_per_vert_hd.unsqueeze(0)
        # calculate com using volume weighted mean
        com = torch.sum(
            vertices_hd * volume_per_vert_hd.unsqueeze(-1), dim=1
        ) / torch.sum(volume_per_vert_hd, dim=1, keepdim=True)

        # get vertices in contact
        ground_plane_height = 0.0
        vertex_height = vertices_hd[:, :, 1] - ground_plane_height
        # vertex_height_robustified = GMoF_unscaled(rho=gmof_rho)(vertex_height)
        contact_mask = (vertex_height < contact_thresh).float()
        num_contact_verts = torch.sum(contact_mask, dim=1)
        contact_metric = torch.ones_like(num_contact_verts)
        contact_metric[torch.nonzero(num_contact_verts == 0)] = 0
        # if torch.nonzero(num_contact_verts).shape[0] != len(num_contact_verts):
        #     print('An element with zero contact vertices found in the batch. Removing that element from stability metric and adding to no_contact metric')
        #     contact_metric[torch.nonzero(num_contact_verts==0)] = 0
        # remove no_contact pose for further computation
        # vertices_hd = torch.index_select(vertices_hd, dim=0, index=torch.nonzero(contact_metric).squeeze())
        # contact_mask = torch.index_select(contact_mask, dim=0, index=torch.nonzero(contact_metric).squeeze())
        # com = torch.index_select(com, dim=0, index=torch.nonzero(contact_metric).squeeze())

        # calculate pressure based cos
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()
        pressure_weights = inside_mask * (
            1 - cop_k * vertex_height
        ) + outside_mask * torch.exp(-cop_w * vertex_height)
        cos = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (
            torch.sum(pressure_weights, dim=1, keepdim=True)
        )

        # get distance of COM to center of support
        # project com, cos to ground plane (x-z plane)
        com_xz = torch.stack([com[:, 0], torch.zeros_like(com)[:, 0], com[:, 2]], dim=1)
        cos_xz = torch.stack([cos[:, 0], torch.zeros_like(cos)[:, 0], cos[:, 2]], dim=1)
        stability_loss = torch.norm(com_xz - cos_xz, dim=1)

    # if non contact, set nan
    stability_loss[torch.nonzero(contact_metric == 0)] = float("nan")
    return stability_loss, contact_metric


def compute_per_part_volume(vertices, faces):
    """
    Compute the volume of each part in the reposed mesh
    """
    with open(SMPL_PART_BOUNDS, "rb") as f:
        part_bounds = {k: pkl.load(f)[k] for k in sorted(pkl.load(f))}

    with open(PART_VID_FID, "rb") as f:
        part_vid_fid = pkl.load(f)

    part_volume = []
    for part_name, part_bounds in part_bounds.items():
        # get part vid and fid
        part_vid = torch.LongTensor(part_vid_fid[part_name]["vert_id"]).to(
            vertices.device
        )
        part_fid = torch.LongTensor(part_vid_fid[part_name]["face_id"]).to(
            vertices.device
        )
        pv = PartVolume(part_name, vertices, faces)
        for bound_name, bound_vids in part_bounds.items():
            pv.close_mesh(bound_vids)
        # add extra vids and fids to original part ids
        new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
        new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
        part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
        part_fid = torch.cat((part_fid, new_face_ids), dim=0)
        pv.extract_part_triangles(part_vid, part_fid)
        part_volume.append(pv.part_volume())
    return torch.vstack(part_volume).permute(1, 0).to(vertices.device)


def face_id_to_part_mapping(part_vid_fid, faces):
    fipm = dict({k: [] for k in range(len(faces))})

    # get mapping from face_id to part label
    for part_label, part_vid_fid in part_vid_fid.items():
        for fid in part_vid_fid["face_id"]:
            if part_label not in fipm[fid]:
                fipm[fid].append(part_label)
    # check if at most two mappings exist. If two, it is a boundary face, take the first one
    for fid, part_labels in fipm.items():
        if len(part_labels) > 2:
            print("Warning: more than two part labels for face {}".format(fid))
            import ipdb

            ipdb.set_trace()
        if len(part_labels) <= 2:
            fipm[fid] = part_labels[0]
    return fipm


class StabilityMetrics:
    def __init__(self, model_type="smpl"):
        if model_type == "smpl":
            num_faces = 13776
            num_verts_hd = 20000

        with open(SMPL_PART_BOUNDS, "rb") as f:
            d = pkl.load(f)
            self.part_bounds = {k: d[k] for k in sorted(d)}
        self.part_order = sorted(self.part_bounds)

        with open(PART_VID_FID, "rb") as f:
            self.part_vid_fid = pkl.load(f)

        # mapping between vid_hd and fid
        with open(HD_SMPL_MAP, "rb") as f:
            faces_vert_is_sampled_from = pkl.load(f)["faces_vert_is_sampled_from"]
        index_row_col = torch.stack(
            [
                torch.LongTensor(np.arange(0, num_verts_hd)),
                torch.LongTensor(faces_vert_is_sampled_from),
            ],
            dim=0,
        )
        values = torch.ones(num_verts_hd, dtype=torch.float)
        size = torch.Size([num_verts_hd, num_faces])
        hd_vert_on_fid = torch.sparse.FloatTensor(index_row_col, values, size)

        # mapping between fid and part label
        with open(FID_TO_PART, "rb") as f:
            fid_to_part_dict = pkl.load(f)
        fid_to_part = torch.zeros(
            [len(fid_to_part_dict.keys()), len(self.part_order)], dtype=torch.float32
        )
        for fid, partname in fid_to_part_dict.items():
            part_idx = self.part_order.index(partname)
            fid_to_part[fid, part_idx] = 1.0

        # mapping between vid_hd and part label
        self.hd_vid_in_part = self.vertex_id_to_part_mapping(
            hd_vert_on_fid, fid_to_part
        )

    def check_bos_interior(
        self, vertices, faces, contact_thresh, no_contact_vert_tol=10
    ):
        """
        Check if the projection of COM lies inside the 2D convex hull/base-of-support of the contact vertices.
        Test this by finding if the COM projectection is a convex combination of the contact vertices.
        Ref: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl#answer-43564754
        Args:
            vertices:
            faces:
            gmof_rho:
            contact_thresh:

        Returns:

        """
        with torch.no_grad():
            batch_size = vertices.shape[0]
            # calculate per part volume
            per_part_volume = self.compute_per_part_volume(vertices, faces)
            # sample 20k vertices uniformly on the smpl mesh
            vertices_hd = HDfier(model_type="smpl").hdfy_mesh(vertices)
            # get volume per vertex id in the hd mesh
            volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(
                per_part_volume, vertices.device
            )
            # calculate com using volume weighted mean
            com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(
                volume_per_vert_hd, dim=1
            )

            # get vertices in contact
            ground_plane_height = 0.0
            vertex_height = vertices_hd[:, :, 1] - ground_plane_height
            # vertex_height_robustified = GMoF_unscaled(rho=gmof_rho)(vertex_height)
            contact_mask = vertex_height < contact_thresh

            num_contact_verts = torch.sum(contact_mask, dim=1, keepdim=True)
            contact_metric = torch.ones_like(num_contact_verts)
            contact_metric[num_contact_verts < no_contact_vert_tol] = 0
            # if torch.nonzero(num_contact_verts).shape[0] != len(num_contact_verts):
            #     print('An element with zero contact vertices found in the batch. Removing that element from stability metric and adding to no_contact metric')
            #     # no_contact_metric is a boolean mask of poses with zero contact vertices
            #     contact_metric = torch.ones_like(num_contact_verts)
            #     contact_metric[torch.nonzero(num_contact_verts == 0)] = 0
            # # remove no_contact pose for further computation
            # vertices_hd = torch.index_select(vertices_hd, dim=0, index=torch.nonzero(contact_metric).squeeze())
            # contact_mask = torch.index_select(contact_mask, dim=0, index=torch.nonzero(contact_metric).squeeze())
            # com = torch.index_select(com, dim=0, index=torch.nonzero(contact_metric).squeeze())

            # filter out the vertices that are not in contact
            # contact_vertices_hd = vertices_hd * contact_mask[:, :, None]
            in_hull_label = torch.zeros(batch_size, dtype=torch.float)
            for i, com_el in enumerate(com):
                # filter out the vertices that are not in contact
                c_el = torch.masked_select(
                    vertices_hd[i], contact_mask[i, :, None]
                ).view(-1, 3)
                if num_contact_verts[i] <= no_contact_vert_tol:
                    in_hull_label[i] = -1  # no contact
                    continue
                # project to ground plane y=0
                c_el[:, 1] = torch.zeros_like(c_el)[:, 1]
                com_el[1] = 0.0
                label = self.in_hull(c_el.cpu().numpy(), com_el.cpu().numpy())
                in_hull_label[i] = float(label)
        return torch.tensor(in_hull_label), com, contact_metric, contact_mask

    def in_hull(self, points, x):
        n_points = len(points)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[points.T, np.ones((1, n_points))]
        b = np.r_[x, np.ones(1)]
        try:
            lp = linprog(c, A_eq=A, b_eq=b, method="interior-point")
            return lp.success
        except:
            print("Linprog failed. Problem is infeasible")
            return False

    def compute_triangle_area(self, triangles):
        ### Compute the area of each triangle in the mesh
        # Compute the cross product of the two vectors of each triangle
        # Then compute the length of the cross product
        # Finally, divide by 2 to get the area of each triangle

        vectors = torch.diff(triangles, dim=2)
        crosses = torch.cross(vectors[:, :, 0], vectors[:, :, 1])
        area = torch.norm(crosses, dim=2) / 2
        return area

    def compute_per_part_volume(self, vertices, faces):
        return compute_per_part_volume(vertices, faces)

    def face_id_to_part_mapping(self, part_vid_fid, faces):
        return face_id_to_part_mapping(part_vid_fid, faces)

    def vertex_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol
