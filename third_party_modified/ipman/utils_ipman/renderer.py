import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
# # hack to get the correct gpu device id on cluster
# #os.environ['PYOPENGL_PLATFORM'] = 'egl'
# #os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]
import torch
from torchvision.utils import make_grid
import cv2
import numpy as np
import pyrender
import matplotlib as mpl
import trimesh
from vis_utils.mesh_utils import get_checkerboard_plane, GMoF_unscaled
from utils.geometry import get_view_matrix
import PIL.Image as pil_img


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self, focal_length=5000, img_res=224, faces=None):

        self.focal_length = focal_length

        if isinstance(img_res, tuple):
            self.camera_center = [img_res[1] // 2, img_res[0] // 2]
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=img_res[1], viewport_height=img_res[0], point_size=1.0
            )
        else:
            # square image
            self.camera_center = [img_res // 2, img_res // 2]
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=img_res, viewport_height=img_res, point_size=1.0
            )
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(
                np.transpose(
                    self.__call__(vertices[i], camera_translation[i], images_np[i]),
                    (2, 0, 1),
                )
            ).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2, alphaMode="OPAQUE", baseColorFactor=(0.8, 0.3, 0.3, 1.0)
        )

        camera_translation[0] *= -1.0

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.camera_center[0],
            cy=self.camera_center[1],
        )
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:, :, None]
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        return output_img


def vis_heatmap(heatmap):
    imgpath = "./data/checkerboard.png"
    image = cv2.imread(imgpath).astype(np.uint8)
    heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    resized_heatmap = cv2.resize(heatmap, (int(image.shape[1]), int(image.shape[0])))
    # idx = resized_heatmap.argmax()

    # center_y = idx // image.shape[1]
    # center_x = idx % image.shape[1]

    colored_heatmap = cv2.applyColorMap(
        resized_heatmap.astype(np.uint8), cv2.COLORMAP_JET
    )
    # colored_heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    masked_image = colored_heatmap * 0.8 + image * 0.2

    # cv2.circle(masked_image, (center_x, center_y), 3, [0, 0, 255], -1)
    # cv2.circle(masked_image, (center_x, center_y), 3, [0, 0, 255], -1)
    return masked_image.astype(np.uint8)


def overlay_mesh(
    verts,
    faces,
    camera_transl,
    focal_length_x,
    focal_length_y,
    camera_center,
    H,
    W,
    img,
    camera_rotation=None,
    rotaround=None,
    topview=False,
    bottomview=False,
    viewer=False,
    gmof_rho=0.01,
    draw_ground_plane=True,
    draw_contact=True,
    draw_support_plane=False,
    com=None,
    xcom=None,
):

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )

    support_plane_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2, alphaMode="BLEND", baseColorFactor=(1.0, 1.0, 1.0, 0.6)
    )

    out_mesh = trimesh.Trimesh(verts, faces, process=False)

    # Set mesh vertex color based on distance from the ground plane
    ground_plane_height = 0.0  # obtained by visualization on the presented pose
    view_offset = 0.0  # groundplane at 0 not visible, lower mesh and ground plane just for viewing

    # Calculate contact vertices
    vertex_height = out_mesh.vertices[:, 1] - ground_plane_height
    vertex_height_robustified = GMoF_unscaled(rho=gmof_rho)(vertex_height)

    if draw_contact:
        cmap = mpl.cm.get_cmap("jet")
        vertex_colors = [cmap(1 - x) for x in vertex_height_robustified]
        out_mesh.visual.vertex_colors = vertex_colors
    else:
        color = np.array([1.00, 0.55, 0.41])  # salmon1
        out_mesh.visual.vertex_colors = color

    # draw ground_plane
    if draw_ground_plane:
        ground_mesh = get_checkerboard_plane()
        ground_pose = np.eye(4)
        if rotaround is not None:
            if topview or bottomview:
                ground_pose = trimesh.transformations.rotation_matrix(
                    np.radians(90), [1, 0, 0]
                )  # make groundplane frontal
            else:
                ground_pose = trimesh.transformations.rotation_matrix(
                    np.radians(0), [1, 0, 0]
                )
            # ground_pose[:3, 3] = np.array([0, ground_plane_height+view_offset, 0])
            # ground_mesh = [mesh.apply_transform(ground_pose) for mesh in ground_mesh]

    # draw support plane that is perpendicular to the ground plane and passes through the centroid of contact vertices
    if draw_support_plane:
        support_mesh = get_checkerboard_plane()
        # Align with y-z plane
        support_pose = trimesh.transformations.rotation_matrix(
            np.radians(90), [0, 1, 0]
        )
        # get z offset
        vertex_idx = vertex_height_robustified < 0.2
        centroid = np.mean(out_mesh.vertices[vertex_idx, :], axis=0)
        support_plane_offset = centroid[
            0
        ]  # ToDo: fit to the contact vertices to minimize distance instead
        support_pose[:3, 3] = np.array([support_plane_offset, 0, 0])

    if rotaround is not None:
        if topview:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(90), [1, 0, 0], [0, 0, 0]
            )
        elif bottomview:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(270), [1, 0, 0], [0, 0, 0]
            )
        else:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rotaround), [0, 1, 0], out_mesh.vertices[4297]
            )
        out_mesh.apply_transform(rot)
        if draw_support_plane:
            support_pose = np.dot(rot, support_pose)
        view_matrix = get_view_matrix(
            pitch=180,
            yaw=-0.5,
            roll=0,
            tx=camera_transl[0],
            ty=camera_transl[1],
            tz=camera_transl[2],
        )
    else:
        camera_pose_gl = get_view_matrix(
            pitch=180 + camera_rotation[0],
            yaw=-camera_rotation[1],
            roll=-camera_rotation[2],
            tx=camera_transl[0],
            ty=-camera_transl[1],
            tz=-camera_transl[2],
        )  # note coversion from camera_orientation to global orientation
        out_mesh.vertices = camera_pose_gl[:3, 3] + np.einsum(
            "mn,jn->jm", camera_pose_gl[:3, :3], verts
        )

        if draw_ground_plane:
            ground_mesh = [mesh.apply_transform(camera_pose_gl) for mesh in ground_mesh]
        if draw_support_plane:
            support_pose = np.dot(camera_pose_gl, support_pose)

        view_matrix = np.eye(4)

    # add origin sphere
    origin_sphere = trimesh.creation.uv_sphere(0.05)
    yellow = [255, 255, 0, 255]
    origin_sphere.visual.face_colors = yellow
    origin_sphere_mesh = pyrender.Mesh.from_trimesh(origin_sphere, smooth=False)
    origin_pose = np.eye(4)

    # add com sphere
    if com is not None:
        com_sphere = trimesh.creation.uv_sphere(0.05)
        green = [0, 255, 0, 255]
        com_sphere.visual.face_colors = green
        com_sphere_mesh = pyrender.Mesh.from_trimesh(com_sphere, smooth=False)
        com_pose = np.eye(4)
        if rotaround is not None:
            origin_pose[:3, 3] = np.array(
                [0, view_offset, 0]
            )  # just for viewing, lower the mesh with the ground plane
            if bottomview:
                com_pose[:3, 3] = np.array([com[0], com[2], 0])
            else:
                com_pose[:3, 3] = np.array(
                    [com[0], 0, com[2]]
                )  # visualize com projection on the ground plane

    # add com sphere
    if xcom is not None:
        xcom_sphere = trimesh.creation.uv_sphere(0.05)
        orange = [255, 99, 71, 255]
        xcom_sphere.visual.face_colors = orange
        xcom_sphere_mesh = pyrender.Mesh.from_trimesh(xcom_sphere, smooth=False)
        xcom_pose = np.eye(4)
        if rotaround is not None:
            origin_pose[:3, 3] = np.array(
                [0, view_offset, 0]
            )  # just for viewing, lower the mesh with the ground plane
            if bottomview:
                xcom_pose[:3, 3] = np.array([xcom[0], xcom[2], 0])
            else:
                xcom_pose[:3, 3] = np.array(
                    [xcom[0], 0, xcom[2]]
                )  # visualize xcom projection on the ground plane

    # add mesh to scene
    mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
    if img is not None:
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3)
        )
    else:
        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(0.3, 0.3, 0.3)
        )
    mesh_pose = np.eye(4)
    if rotaround is not None:
        mesh_pose[:3, 3] = np.array(
            [0, view_offset, 0]
        )  # just for viewing, lower the mesh with the ground plane
    scene.add(mesh, pose=mesh_pose, name="mesh")

    if draw_ground_plane:
        ground_mesh = pyrender.Mesh.from_trimesh(ground_mesh, smooth=False)
        # scene.add(ground_mesh, pose=ground_pose, name='ground_plane')
        scene.add(ground_mesh, name="ground_plane")

    if draw_support_plane:
        support_mesh = [mesh.apply_transform(support_pose) for mesh in support_mesh]
        support_mesh = pyrender.Mesh.from_trimesh(
            support_mesh, smooth=False, material=support_plane_material
        )
        scene.add(support_mesh, name="support_plane")

    scene.add(origin_sphere_mesh, pose=origin_pose, name="origin_sphere")
    if com is not None:
        scene.add(com_sphere_mesh, pose=com_pose, name="com_sphere")
    if xcom is not None:
        scene.add(xcom_sphere_mesh, pose=xcom_pose, name="xcom_sphere")

    # add mesh for camera
    if img is None and topview is False:
        # to lower the ground-plane so person doesn't get cut
        cam_y_offset = 150
    elif topview:
        cam_y_offset = -100
    else:
        cam_y_offset = 0
    pyrencamera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length_x,
        fy=focal_length_y,
        cx=camera_center[0],
        cy=camera_center[1] + cam_y_offset,
    )
    scene.add(pyrencamera, pose=view_matrix)

    # create and add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)
    for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
        light_pose[:3, 3] = out_mesh.vertices.mean(0) + np.array(lp)
        # out_mesh.vertices.mean(0) + np.array(lp)
        scene.add(light, pose=light_pose)

    if viewer:
        pyrender.Viewer(scene, use_raymond_lighting=True)
        return 0
    else:
        r = pyrender.OffscreenRenderer(
            viewport_width=W, viewport_height=H, point_size=1.0
        )
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if img is not None:
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img
        else:
            output_img = color

        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]

        return output_img
