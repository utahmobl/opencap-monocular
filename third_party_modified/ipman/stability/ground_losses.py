# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import sys
import os

import torch
import torch.nn as nn
import pickle as pkl
import numpy as np

ipman_dir = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(ipman_dir)

from ..vis_utils.mesh_utils import HDfier
from ..utils_ipman.geometry import ea2rm
from ..constants import CONTACT_THRESH
from ..stability.part_volumes import PartVolume

SMPL_PART_BOUNDS = os.path.join(
    ipman_dir,
    "data/essentials/yogi_segments/smpl/part_meshes_ply/smpl_segments_bounds.pkl",
)
FID_TO_PART = os.path.join(
    ipman_dir, "data/essentials/yogi_segments/smpl/part_meshes_ply/fid_to_part.pkl"
)
PART_VID_FID = os.path.join(
    ipman_dir,
    "data/essentials/yogi_segments/smpl/part_meshes_ply/smpl_part_vid_fid.pkl",
)
HD_SMPL_MAP = os.path.join(
    ipman_dir, "data/essentials/hd_model/smpl/smpl_neutral_hd_sample_from_mesh_out.pkl"
)


class StabilityLossCoP(nn.Module):
    def __init__(
        self,
        faces,
        cop_w=10,
        cop_k=100,
        contact_thresh=CONTACT_THRESH,
        model_type="smpl",
        device="cuda",
    ):
        super().__init__()
        """
        Loss that ensures that the COM of the SMPL mesh is close to the center of support 
        """
        if model_type == "smpl":
            num_faces = 13776
            num_verts_hd = 20000

        self.device = device

        assert faces is not None, "Faces tensor is none"
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long).to(device)
        self.register_buffer("faces", faces)

        self.cop_w = cop_w
        self.cop_k = cop_k
        self.contact_thresh = contact_thresh

        self.hdfy_op = HDfier(model_type=model_type)

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
        ).to(device)
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

    def compute_triangle_area(self, triangles):
        ### Compute the area of each triangle in the mesh
        # Compute the cross product of the two vectors of each triangle
        # Then compute the length of the cross product
        # Finally, divide by 2 to get the area of each triangle

        vectors = torch.diff(triangles, dim=2)
        crosses = torch.cross(vectors[:, :, 0], vectors[:, :, 1])
        area = torch.norm(crosses, dim=2) / 2
        return area

    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]["vert_id"]).to(
                vertices.device
            )
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]["face_id"]).to(
                vertices.device
            )
            pv = PartVolume(part_name, vertices, self.faces)
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

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol

    def vertex_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part

    def compute_com(self, vertices):
        # assuming SMPL orientation of z up, rotate the vertices to align with the pyrender coordinates (y-up)
        R1 = (
            ea2rm(
                torch.tensor([[np.radians(270)]]),
                torch.tensor([[np.radians(0)]]),
                torch.tensor([[np.radians(0)]]),
            )
            .float()
            .to(self.device)
            .expand(vertices.shape[0], 3, 3)
        )
        vertices = torch.einsum("bki,bji->bjk", [R1, vertices])

        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(
            per_part_volume, vertices.device
        )
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(
            volume_per_vert_hd, dim=1
        )

        # rotate back to original
        R_to_smpl = (
            ea2rm(
                torch.tensor([[np.radians(-270)]]),
                torch.tensor([[np.radians(0)]]),
                torch.tensor([[np.radians(0)]]),
            )
            .float()
            .to(self.device)
            .expand(vertices.shape[0], 3, 3)
        )
        com_rotated_smpl = torch.einsum("bji,bi->bj", [R_to_smpl, com])

        return com_rotated_smpl

    def forward(self, vertices):
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(
            per_part_volume, vertices.device
        )
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(
            volume_per_vert_hd, dim=1
        )

        # # get COM of the SMPLX mesh
        # triangles = torch.index_select(vertices, 1, self.faces.view(-1)).reshape(batch_size, -1, 3, 3)
        # triangle_centroids = torch.mean(triangles, dim=2)
        # triangle_area = self.compute_triangle_area(triangles)
        # com_naive = torch.einsum('bij,bi->bj', triangle_centroids, triangle_area) / torch.sum(triangle_area, dim=1)

        # pressure based center of support
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = vertices_hd[:, :, 1] - ground_plane_height
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()
        pressure_weights = inside_mask * (
            1 - self.cop_k * vertex_height
        ) + outside_mask * torch.exp(-self.cop_w * vertex_height)
        cop = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (
            torch.sum(pressure_weights, dim=1, keepdim=True) + eps
        )

        # naive center of support
        # vertex_height_robustified = GMoF_unscaled(rho=self.gmof_rho)(vertex_height)
        contact_confidence = torch.sum(pressure_weights, dim=1)
        # contact_mask = (vertex_height < self.contact_thresh).float()
        # num_contact_verts = torch.sum(contact_mask, dim=1)
        # contact_centroid_naive = torch.sum(vertices_hd * contact_mask[:, :, None], dim=1) / (torch.sum(contact_mask, dim=1) + eps)

        # project com, cop to ground plane (x-z plane)
        # weight loss by number of contact vertices to zero out if zero vertices in contact
        com_xz = torch.stack([com[:, 0], torch.zeros_like(com)[:, 0], com[:, 2]], dim=1)
        contact_centroid_xz = torch.stack(
            [cop[:, 0], torch.zeros_like(cop)[:, 0], cop[:, 2]], dim=1
        )
        # stability_loss = (contact_confidence * torch.norm(com_xz - contact_centroid_xz, dim=1)).sum(dim=-1)
        stability_loss = torch.norm(com_xz - contact_centroid_xz, dim=1)
        return stability_loss


class GroundLoss(nn.Module):
    def __init__(
        self,
        faces,
        device="cuda",
        model_type="smpl",
        out_alpha1=1,
        out_alpha2=0.5,
        in_alpha1=1,
        in_alpha2=0.15,
    ):
        super().__init__()
        assert faces is not None, "Faces tensor is none"
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long).to(device)
        self.register_buffer("faces", faces)

        self.device = device

        # loss specifications

        # push, pull, and contact weights
        # self.inside_w = inside_loss_weight
        # self.outside_w = outside_loss_weight

        # hyper params
        self.a1 = out_alpha1
        self.a2 = out_alpha2
        self.b1 = in_alpha1
        self.b2 = in_alpha2

        self.hdfy_op = HDfier(model_type=model_type)

    def forward(self, vertices):
        """
        Loss that ensures that the ground plane pulls body mesh vertices towards it till contact and resolves
        ground-plane intersections
        Using assymetric loss such that the
        - loss is 0 when the body is in contact with the ground plane
        - loss is >> 0 when the plane interpenetrates the body (completely implausible)
        - loss >0 if the plane is close to nearby vertices that are not touching
        vertices:the vertices should be aligned along y-axis and in world coordinates
        """
        bs = vertices.shape[0]
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get vertices under the ground plane
        ground_plane_height = 0.0  # obtained by visualization on the presented pose
        vertex_height = vertices_hd[:, :, 1] - ground_plane_height
        inside_mask = vertex_height < 0.00
        outside_mask = vertex_height >= 0.00
        # pull closeby outside vertices
        v2v_pull = self.a1 * torch.tanh((vertex_height * outside_mask) / self.a2) ** 2
        # # apply loss to inside vertices to remove intersection
        v2v_push = self.b1 * torch.tanh((vertex_height * inside_mask) / self.b2) ** 2
        return v2v_pull, v2v_push


# class MimickedSupportLoss(nn.Module):
#     def __init__(self,
#                  faces,
#                  inside_loss_weight=1.0,
#                  outside_loss_weight=1.0,
#                  contact_loss_weight=2.5,
#                  align_faces=False,
#                  gmof_rho=0.02,
#                  use_hd=True,
#                  device='cuda',
#                  model_type='smplx',
#                  alpha1=0.005,
#                  alpha2=0.005,
#                  beta1=600,
#     ):
#         super().__init__()
#         assert faces is not None, 'Faces tensor is none'
#         if type(faces) is not torch.Tensor:
#             faces = torch.tensor(faces.astype(np.int64), dtype=torch.long)
#         self.register_buffer('faces', faces)
#
#         self.device = device
#         self.model_type = model_type
#
#         # loss specifications
#         self.use_hd = use_hd
#         self.align_faces = align_faces
#         # self.gmof_rho = gmof_rho
#
#         # push, pull, and contact weights
#         self.inside_w = inside_loss_weight
#         self.contact_w = contact_loss_weight
#
#         # hyper configs
#         self.a1 = alpha1
#         self.a2 = alpha2
#         self.b1 = beta1
#
#
#
#     def forward(self, vertices):
#         """
#                 Loss that ensures that the ground plane pulls body mesh vertices towards it till contact and resolves
#                 ground-plane intersections
#                 Using assymetric loss such that the
#                 - loss is 0 when the body is in contact with the ground plane
#                 - loss is >> 0 when the plane interpenetrates the body (completely implausible)
#                 - loss >0 if the plane is close to nearby vertices that are not touching
#         """
#         bs = vertices.shape[0]
#
#         import ipdb; ipdb.set_trace()
#         # get vertices under the ground plane
#         ground_plane_height = 0.0  # obtained by visualization on the presented pose
#         vertex_height = (vertices[:, :, 1] - ground_plane_height)
#         inside_mask = (vertex_height < 0.00).float()
#
#         # pull closeby outside vertices
#         v2v_pull = self.contact_w * (self.a1 * torch.tanh(vertex_height * [~inside_mask] / self.a2)**2).sum()
#         # # apply loss to inside vertices to remove intersection
#         v2v_push = self.inside_w * (self.b1 * (vertex_height * inside_mask)**2).sum()
#         contactloss = v2v_push + v2v_pull
#
#         # now align faces that are close
#         # dot product should be -1 for faces in contact
#         if self.align_faces:
#             triangles = torch.index_select(vertices, 1, self.faces.view(-1)).reshape(bs, -1, 3, 3)
#             face_normals = batch_face_normals(triangles)
#
#         face_angle_loss = torch.zeros(bs, device=vertices.device)
#         if self.align_faces:
#             if self.use_hd:
#                 if hd_faces_in_contact[idx] is not None:
#                     hd_fn_in_contact = face_normals[idx][hd_faces_in_contact[idx]]
#                     dotprod_normals = 1 + (hd_fn_in_contact[0] * hd_fn_in_contact[1]).sum(1)
#                     face_angle_loss[idx] = dotprod_normals.sum()
#             else:
#                 sys.exit('You can only align vertices when use_hd=True')
#
#         return contactloss, face_angle_loss
