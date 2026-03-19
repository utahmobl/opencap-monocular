# borrowed from https://github.com/CZ-Wu/GPNet/blob/d8df3b1489800626d4f503f62d2ae8daffe8b686/tools/visualization.py

import trimesh
import os.path as osp
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from vis_utils.constants import ESSENTIALS_DIR
import torch.nn as nn


def batch_face_normals(triangles):
    # Calculate the edges of the triangles
    # Size: BxFx3
    edge0 = triangles[:, :, 1] - triangles[:, :, 0]
    edge1 = triangles[:, :, 2] - triangles[:, :, 0]
    # Compute the cross product of the edges to find the normal vector of
    # the triangle
    aCrossb = torch.cross(edge0, edge1, dim=2)
    # Normalize the result to get a unit vector
    normals = aCrossb / torch.norm(aCrossb, 2, dim=2, keepdim=True)

    return normals


def sparse_batch_mm(m1, m2):
    """
    https://github.com/pytorch/pytorch/issues/14489

    m1: sparse matrix of size N x M
    m2: dense matrix of size B x M x K
    returns m1@m2 matrix of size B x N x K
    """

    batch_size = m2.shape[0]
    # stack m2 into columns: (B x N x K) -> (N, B, K) -> (N, B * K)
    m2_stack = m2.transpose(0, 1).reshape(m1.shape[1], -1)
    result = m1.mm(m2_stack).reshape(m1.shape[0], batch_size, -1).transpose(1, 0)
    return result


class HDfier:
    def __init__(self, model_type="smpl", device="cuda"):
        hd_operator_path = osp.join(
            ESSENTIALS_DIR,
            "hd_model",
            model_type,
            f"{model_type}_neutral_hd_vert_regressor_sparse.npz",
        )
        hd_operator = np.load(hd_operator_path)
        self.hd_operator = torch.sparse_coo_tensor(
            torch.tensor(hd_operator["index_row_col"]),
            torch.tensor(hd_operator["values"]),
            torch.Size(hd_operator["size"]),
        )
        self.model_type = model_type
        self.device = device

    def hdfy_mesh(self, vertices):
        """
        Applies a regressor that maps SMPL vertices to uniformly distributed vertices
        """
        # device = body.vertices.device
        # check if vertices ndim are 3, if not , add a new axis
        if vertices.dim() != 3:
            # batchify the vertices
            vertices = vertices[None, :, :]

        # check if vertices are an ndarry, if yes, make pytorch tensor
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(self.device)

        vertices = vertices.to(torch.double)

        if self.hd_operator.device != vertices.device:
            self.hd_operator = self.hd_operator.to(vertices.device)
        hd_verts = sparse_batch_mm(self.hd_operator, vertices).to(torch.float)
        return hd_verts


def get_submesh(
    verts, faces, verts_retained=None, faces_retained=None, min_vert_in_face=2
):
    """
    Given a mesh, create a (smaller) submesh
    indicate faces or verts to retain as indices or boolean

    @return new_verts: the new array of 3D vertices
            new_faces: the new array of faces
            bool_faces: the faces indices wrt the input mesh
            vetex_ids: the vertex_ids wrt the input mesh
    """

    if verts_retained is not None:
        # Transform indices into bool array
        if verts_retained.dtype != "bool":
            vert_mask = np.zeros(len(verts), dtype=bool)
            vert_mask[verts_retained] = True
        else:
            vert_mask = verts_retained

        # Faces with at least min_vert_in_face vertices
        bool_faces = (
            np.sum(vert_mask[faces.ravel()].reshape(-1, 3), axis=1) > min_vert_in_face
        )

    elif faces_retained is not None:
        # Transform indices into bool array
        if faces_retained.dtype != "bool":
            bool_faces = np.zeros(len(faces_retained), dtype=bool)
        else:
            bool_faces = faces_retained

    new_faces = faces[bool_faces]
    # just in case additional vertices are added
    vertex_ids = list(set(new_faces.ravel()))

    oldtonew = -1 * np.ones([len(verts)])
    oldtonew[vertex_ids] = range(0, len(vertex_ids))

    new_verts = verts[vertex_ids]
    new_faces = oldtonew[new_faces].astype("int32")

    return (new_verts, new_faces, bool_faces, vertex_ids)


def get_world_mesh_list(planeWidth=4, axisHeight=0.7, axisRadius=0.02, add_plane=True):
    groundColor = [220, 220, 220, 255]  # face_colors: [R, G, B, transparency]
    xColor = [255, 0, 0, 128]
    yColor = [0, 255, 0, 128]
    zColor = [0, 0, 255, 128]

    if add_plane:
        ground = trimesh.primitives.Box(
            center=[0, 0, -0.0001], extents=[planeWidth, planeWidth, 0.0002]
        )
        ground.visual.face_colors = groundColor

    xAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    xAxis.apply_transform(
        matrix=np.mat(
            ((0, 0, 1, axisHeight / 2), (0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 0, 1))
        )
    )
    xAxis.visual.face_colors = xColor
    yAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    yAxis.apply_transform(
        matrix=np.mat(
            ((1, 0, 0, 0), (0, 0, -1, axisHeight / 2), (0, 1, 0, 0), (0, 0, 0, 1))
        )
    )
    yAxis.visual.face_colors = yColor
    zAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    zAxis.apply_transform(
        matrix=np.mat(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, axisHeight / 2), (0, 0, 0, 1))
        )
    )
    zAxis.visual.face_colors = zColor
    xBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    xBox.apply_translation((axisHeight, 0, 0))
    xBox.visual.face_colors = xColor
    yBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    yBox.apply_translation((0, axisHeight, 0))
    yBox.visual.face_colors = yColor
    zBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    zBox.apply_translation((0, 0, axisHeight))
    zBox.visual.face_colors = zColor
    if add_plane:
        worldMeshList = [ground, xAxis, yAxis, zAxis, xBox, yBox, zBox]
    else:
        worldMeshList = [xAxis, yAxis, zAxis, xBox, yBox, zBox]
    return worldMeshList


def get_checkerboard_plane(plane_width=5, num_boxes=9, center=True):

    pw = plane_width / num_boxes
    white = [220, 220, 220, 255]
    black = [35, 35, 35, 255]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            ground = trimesh.primitives.Box(
                center=[0, 0, -0.0001], extents=[pw, pw, 0.0002]
            )
            if center:
                c = c[0] + (pw / 2) - (plane_width / 2), c[1] + (pw / 2) - (
                    plane_width / 2
                )
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.apply_translation([c[0], c[1], 0])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.visual.face_colors = black if ((i + j) % 2) == 0 else white
            meshes.append(ground)

    # orient mesh be ground in pyrender axis system
    pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    meshes = [mesh.apply_transform(pose) for mesh in meshes]
    return meshes


def get_meshed_plane(num_boxes=9):
    def makeVertexGrid(n, m):
        # vertices are incremented in X and Y but remain at zero in Z
        x = np.repeat(np.arange(-n / 2, n / 2), m)
        y = np.tile(np.arange(-m / 2, m / 2), n)
        z = np.zeros(n * m)
        grid = np.column_stack((x, y, z))
        return grid

    def makeFaceGrid(n, m):
        # Each face references three vertices by their index numbers, in counterclockwise order. for every vertex i not
        # in the right column or the top row, triangles [i, i+m,i+m+1] and [i, i+m+1, i+1] are created, where m is the
        # number of rows in the array.
        grid = np.zeros((2 * (m - 1) * (n - 1), 3))
        counter = 0
        for i in range((n - 1) * m):
            if i % m == m - 1:
                pass
            else:
                grid[counter] = [i, i + m, i + m + 1]
                grid[counter + 1] = [i, i + m + 1, i + 1]
                counter += 2
        return grid

    vertices = makeVertexGrid(num_boxes, num_boxes)
    faces = makeFaceGrid(num_boxes, num_boxes)

    # Add faces at the top of the mesh to avoid backface culling during visualization
    faces = np.vstack((faces, np.fliplr(faces)))

    white = [220, 220, 220, 255]
    black = [35, 35, 35, 255]
    face_colors = [black if i % 2 == 0 else white for i, _ in enumerate(faces)]
    ground = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)
    # ground.visual.face_colors = white
    return ground


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return "rho = {}".format(self.rho)

    def forward(self, residual):
        squared_res = residual**2
        dist = torch.div(squared_res, squared_res + self.rho**2)
        return self.rho**2 * dist


class GMoF_unscaled(nn.Module):
    def __init__(self, rho=1):
        super(GMoF_unscaled, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return "rho = {}".format(self.rho)

    def forward(self, residual):
        squared_res = residual**2
        if type(residual) is torch.Tensor:
            dist = torch.div(squared_res, squared_res + self.rho**2)
        else:
            dist = squared_res / (squared_res + self.rho**2)
        return dist


def getNewCoordinate(axisHeight=0.05, axisRadius=0.001):
    xColor = [200, 50, 0, 128]
    yColor = [0, 200, 50, 128]
    zColor = [50, 0, 200, 128]

    xAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    xAxis2.apply_transform(
        matrix=np.mat(
            ((0, 0, 1, axisHeight / 2), (0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 0, 1))
        )
    )
    xAxis2.visual.face_colors = xColor
    yAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    yAxis2.apply_transform(
        matrix=np.mat(
            ((1, 0, 0, 0), (0, 0, -1, axisHeight / 2), (0, 1, 0, 0), (0, 0, 0, 1))
        )
    )
    yAxis2.visual.face_colors = yColor
    zAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    zAxis2.apply_transform(
        matrix=np.mat(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, axisHeight / 2), (0, 0, 0, 1))
        )
    )
    zAxis2.visual.face_colors = zColor
    xBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    xBox2.apply_translation((axisHeight, 0, 0))
    xBox2.visual.face_colors = xColor
    yBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    yBox2.apply_translation((0, axisHeight, 0))
    yBox2.visual.face_colors = yColor
    zBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    zBox2.apply_translation((0, 0, axisHeight))
    zBox2.visual.face_colors = zColor

    return 1


def meshVisualization(mesh):
    worldMeshList = get_world_mesh_list(
        planeWidth=0.2, axisHeight=0.05, axisRadius=0.001
    )
    mesh.visual.face_colors = [255, 128, 255, 200]
    worldMeshList.append(mesh)
    scene = trimesh.Scene(worldMeshList)
    scene.show()


def meshPairVisualization(mesh1, mesh2):
    worldMeshList = get_world_mesh_list(
        planeWidth=0.2, axisHeight=0.05, axisRadius=0.001
    )
    mesh1.visual.face_colors = [255, 128, 255, 200]
    mesh2.visual.face_colors = [255, 255, 128, 200]

    worldMeshList.append((mesh1, mesh2))
    scene = trimesh.Scene(worldMeshList)
    scene.show()


if __name__ == "__main__":
    # meshVisualization(trimesh.convex.convex_hull(np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]])))
    scene = trimesh.Scene(get_checkerboard_plane())
    scene.add_geometry(get_world_mesh_list(add_plane=False))
    scene.show()
