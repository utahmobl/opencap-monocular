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
"""
This script is used to close part meshes.
"""
from tempfile import gettempdir

import torch
import trimesh
import torch.nn as nn
import numpy as np
import pickle

# from .utils.mesh import winding_numbers


class SMPLMesh(nn.Module):
    def __init__(self, vertices, faces):
        super(SMPLMesh, self).__init__()

        self.vertices = vertices
        self.faces = faces


class PartVolume(nn.Module):
    def __init__(self, part_name, vertices, faces):
        super(PartVolume, self).__init__()

        self.part_name = part_name
        self.smpl_mesh = SMPLMesh(vertices, faces)

        self.part_triangles = None
        self.device = vertices.device

        self.new_vert_ids = []
        self.new_face_ids = []

    def close_mesh(self, boundary_vids):
        # find the center of the boundary
        mean_vert = self.smpl_mesh.vertices[:, boundary_vids, :].mean(
            dim=1, keepdim=True
        )
        self.smpl_mesh.vertices = torch.cat([self.smpl_mesh.vertices, mean_vert], dim=1)
        new_vert_idx = self.smpl_mesh.vertices.shape[1] - 1
        self.new_vert_ids.append(new_vert_idx)
        # add faces
        new_faces = [
            [boundary_vids[i + 1], boundary_vids[i], new_vert_idx]
            for i in range(len(boundary_vids) - 1)
        ]
        self.smpl_mesh.faces = torch.cat(
            [
                self.smpl_mesh.faces,
                torch.tensor(new_faces, dtype=torch.long, device=self.device),
            ],
            dim=0,
        )
        self.new_face_ids += list(
            range(
                self.smpl_mesh.faces.shape[0] - len(new_faces),
                self.smpl_mesh.faces.shape[0],
            )
        )

    def extract_part_triangles(self, part_vids, part_fid):
        # make sure that inputs are from a watertight part mesh
        batch_size = self.smpl_mesh.vertices.shape[0]

        part_vertices = self.smpl_mesh.vertices[:, part_vids, :]
        part_faces = self.smpl_mesh.faces[part_fid, :]

        part_mean = part_vertices.mean(dim=1, keepdim=True)

        # subtract vert mean because volume computation only applies if origin is inside the triangles
        self.smpl_mesh.vertices = self.smpl_mesh.vertices - part_mean

        # compute triangle
        if self.part_triangles is None:
            # self.part_triangles = torch.index_select(self.smpl_mesh.vertices, 1, self.smpl_mesh.faces.view(-1)).reshape(batch_size, -1, 3, 3)
            self.part_triangles = torch.index_select(
                self.smpl_mesh.vertices, 1, part_faces.view(-1)
            ).reshape(batch_size, -1, 3, 3)
        else:
            self.part_triangles = torch.cat(
                [
                    self.part_triangles,
                    torch.index_select(
                        self.smpl_mesh.vertices, 1, part_faces.view(-1)
                    ).reshape(batch_size, -1, 3, 3),
                ],
                dim=1,
            )
        # add back vert mean
        self.smpl_mesh.vertices = self.smpl_mesh.vertices + part_mean

    def part_volume(self):
        # Note: the mesh should be enclosing the origin (mean-subtracted)
        # compute volume of triangles by drawing tetrahedrons
        # https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up
        x = self.part_triangles[:, :, :, 0]
        y = self.part_triangles[:, :, :, 1]
        z = self.part_triangles[:, :, :, 2]
        volume = (
            -x[:, :, 2] * y[:, :, 1] * z[:, :, 0]
            + x[:, :, 1] * y[:, :, 2] * z[:, :, 0]
            + x[:, :, 2] * y[:, :, 0] * z[:, :, 1]
            - x[:, :, 0] * y[:, :, 2] * z[:, :, 1]
            - x[:, :, 1] * y[:, :, 0] * z[:, :, 2]
            + x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
        ).sum(dim=1).abs() / 6.0
        return volume


class BodySegment(nn.Module):
    def __init__(
        self, name, faces, segments_folder, model_type="smpl", append_idx=None
    ):
        super(BodySegment, self).__init__()

        self.name = name
        self.append_idx = faces.max().item() if append_idx is None else append_idx

        self.model_type = model_type
        sb_path = f"{segments_folder}/{model_type}_segments_bounds.pkl"
        sxseg = pickle.load(open(sb_path, "rb"))

        # read mesh and find faces of segment
        segment_path = f"{segments_folder}/{model_type}_segment_{name}.ply"
        bandmesh = trimesh.load(segment_path, process=False)
        segment_vidx = torch.from_numpy(
            np.where(np.array(bandmesh.visual.vertex_colors[:, 0]) == 255)[0]
        )
        self.register_buffer("segment_vidx", segment_vidx)

        # read boundary information
        self.bands = [x for x in sxseg[name].keys()]
        self.bands_verts = [x for x in sxseg[name].values()]
        self.num_bounds = len(self.bands_verts)
        for idx, bv in enumerate(self.bands_verts):
            self.register_buffer(f"bands_verts_{idx}", torch.tensor(bv))
        self.bands_faces = self.create_band_faces()

        # read mesh and find
        segment_faces_ids = np.where(
            np.isin(faces.cpu().numpy(), segment_vidx).sum(1) == 3
        )[0]
        segment_faces = faces[segment_faces_ids, :]
        segment_faces = torch.cat((faces[segment_faces_ids, :], self.bands_faces), 0)
        self.register_buffer("segment_faces", segment_faces)

        # create vector to select vertices form faces
        tri_vidx = []
        for ii in range(faces.max().item() + 1):
            tri_vidx += [torch.nonzero(faces == ii)[0].tolist()]
        self.register_buffer("tri_vidx", torch.tensor(tri_vidx))

    def create_band_faces(self):
        """
        create the faces that close the segment.
        """
        bands_faces = []
        for idx, k in enumerate(self.bands):
            new_vert_idx = self.append_idx + 1 + idx
            new_faces = [
                [self.bands_verts[idx][i + 1], self.bands_verts[idx][i], new_vert_idx]
                for i in range(len(self.bands_verts[idx]) - 1)
            ]
            bands_faces += new_faces

        bands_faces_tensor = torch.tensor(
            np.array(bands_faces).astype(np.int64), dtype=torch.long
        )

        return bands_faces_tensor

    def get_closed_segment(self, vertices):
        """
        create the closed segment mesh from SMPL-X vertices.
        """
        vertices = vertices.detach().clone()
        # append vertices to smpl, that close the segment and compute faces
        for idx in range(self.num_bounds):
            bv = eval(f"self.bands_verts_{idx}")
            close_segment_vertices = torch.mean(vertices[:, bv, :], 1, keepdim=True)
            vertices = torch.cat((vertices, close_segment_vertices), 1)
        segm_triangles = vertices[:, self.segment_faces, :]

        return segm_triangles

    # def has_self_isect_verts(self, vertices, thres=0.99):
    #     """
    #         check if segment (its vertices) are self intersecting.
    #     """
    #     segm_triangles = self.get_closed_segment(vertices)
    #     segm_verts = vertices[:,self.segment_vidx,:]
    #
    #     # do inside outside segmentation
    #     exterior = winding_numbers(segm_verts, segm_triangles) \
    #                 .le(thres)
    #
    #     return exterior
    #
    # def has_self_isect_points(self, points, triangles, thres=0.99):
    #     """
    #         check if points on segment are self intersecting.
    #     """
    #     smpl_verts = triangles[:,self.tri_vidx[:,0], self.tri_vidx[:,1],:]
    #     segm_triangles = self.get_closed_segment(smpl_verts)
    #
    #     # do inside outside segmentation
    #     exterior = winding_numbers(points, segm_triangles) \
    #                 .le(thres)
    #
    #     return exterior


class BatchBodySegment(nn.Module):
    def __init__(self, names, faces, segments_folder, model_type="smpl", device="cuda"):
        super(BatchBodySegment, self).__init__()
        self.names = names
        self.num_segments = len(names)
        self.nv = faces.max().item()

        self.model_type = model_type
        sb_path = f"{segments_folder}/{model_type}_segments_bounds.pkl"
        sxseg = pickle.load(open(sb_path, "rb"))

        self.append_idx = [
            len(b) for a, b in sxseg.items() for c, d in b.items() if a in self.names
        ]
        self.append_idx = np.cumsum(np.array([self.nv] + self.append_idx))

        self.segmentation = {}
        for idx, name in enumerate(names):
            self.segmentation[name] = BodySegment(
                name, faces, segments_folder, model_type
            ).to("cuda")

    def batch_has_self_isec_verts(self, vertices):
        """
        check is mesh is intersecting with itself
        """
        exteriors = []
        for k, segm in self.segmentation.items():
            exteriors += [segm.has_self_isect_verts(vertices)]
        return exteriors
