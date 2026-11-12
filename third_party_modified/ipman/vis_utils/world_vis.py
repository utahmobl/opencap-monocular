import os
import os.path as osp
import torch
import trimesh
import pyrender
import numpy as np
import matplotlib as mpl
import cv2
import pickle as pkl

import PIL.Image as pil_img
from PIL import ImageFont
from PIL import ImageDraw
from constants import CONTACT_THRESH, GROUND_OFFSETS
from vis_utils.mesh_utils import (
    get_checkerboard_plane,
    get_meshed_plane,
    GMoF_unscaled,
    HDfier,
)

# from lib.core.part_volumes import PartVolume
from utils.geometry import ea2rm
from stability.stability_metrics import StabilityMetrics

# from psbody.mesh.colors import name_to_rgb
os.environ["PYOPENGL_PLATFORM"] = "egl"

from smplx import SMPL

SMPL_MODEL_DIR = "/ps/project/alignment/models/smplx_internal/smpl/"
SMPL_BATCH_SIZE = 500
smpl_obj = SMPL(SMPL_MODEL_DIR, batch_size=SMPL_BATCH_SIZE, create_transl=True)

STABILITY_METRICS = StabilityMetrics()


def vis_vert_with_ground(
    vertices,
    trans,
    seq_name,
    vis_path="dummy",
    device="cuda",
    start_idx=0,
    sub_sample=5,
    ground_offset=GROUND_OFFSETS["h36m"],
    contact_thresh=CONTACT_THRESH,
    imgnames=[],
    smpl_batch_size=SMPL_BATCH_SIZE,
    visualize=True,
):
    global smpl_obj
    smpl_obj = smpl_obj.to(device)
    with torch.no_grad():
        vertices += trans
        cam_rot = trimesh.transformations.rotation_matrix(np.radians(-30), [1, 0, 0])
        img_count = start_idx
        out_dir = os.path.join(vis_path, seq_name)
        os.makedirs(out_dir, exist_ok=True)
        faces = torch.tensor(smpl_obj.faces.astype(np.int64), dtype=torch.long).to(
            device
        )
        # rotate the vertices to align with the pyrender coordinates, no aligned with y axis
        R1 = (
            ea2rm(
                torch.tensor([[np.radians(270)]]),
                torch.tensor([[np.radians(0)]]),
                torch.tensor([[np.radians(0)]]),
            )
            .float()
            .to(device)
            .expand(vertices.shape[0], 3, 3)
        )
        vertices = torch.einsum("bki,bji->bjk", [R1, vertices])

        vertices[:, :, 1] += -ground_offset
        in_bos_label, com, contact_metric, contact_mask = (
            STABILITY_METRICS.check_bos_interior(
                vertices, faces, contact_thresh=contact_thresh
            )
        )

        for fnum in range(min(smpl_batch_size // sub_sample, len(vertices))):
            # check stability (com inside bos)
            if visualize:
                # camera values come from visual inspection in yogi/smplify-xmc-support/rendering
                view1 = overlay_mesh(
                    vertices[fnum].cpu().numpy(),
                    smpl_obj.faces,
                    camera_transl=np.array([-1.7, 2, 7]),
                    # camera_transl=np.array([-0, 2, 6]), # for last two sequences in rich test
                    focal_length_x=1000,
                    focal_length_y=1000,
                    camera_center=[150, 250],
                    H=1000,
                    W=1000,
                    img=None,
                    camera_rotation=cam_rot,
                    rotaround=0,
                    com=com.cpu().numpy(),
                    gmof_rho=contact_thresh,
                )
                # if img_count == 396:
                view2 = overlay_mesh(
                    vertices[fnum].cpu().numpy(),
                    smpl_obj.faces,
                    camera_transl=np.array([-1.7, 2, 5]),
                    # camera_transl=np.array([-0, 2, 6]), # for last two sequences in rich test
                    focal_length_x=1000,
                    focal_length_y=1000,
                    camera_center=[150, 250],
                    H=1000,
                    W=1000,
                    img=None,
                    camera_rotation=cam_rot,
                    rotaround=45,
                    com=com.cpu().numpy(),
                    gmof_rho=contact_thresh,
                )
                view3 = overlay_mesh(
                    vertices[fnum].cpu().numpy(),
                    smpl_obj.faces,
                    camera_transl=np.array([-1.7, 2, 5]),
                    # camera_transl=np.array([-0, 2, 6]), # for last two sequences in rich test
                    focal_length_x=1000,
                    focal_length_y=1000,
                    camera_center=[150, 250],
                    H=1000,
                    W=1000,
                    img=None,
                    camera_rotation=cam_rot,
                    rotaround=90,
                    com=com.cpu().numpy(),
                    gmof_rho=contact_thresh,
                )
                img_bottom = overlay_mesh(
                    vertices[fnum].cpu().numpy(),
                    smpl_obj.faces,
                    camera_transl=np.array([-1.7, 2, 7]),
                    # camera_transl=np.array([-0, 2, 7]), # for last two sequences in rich test
                    focal_length_x=1000,
                    focal_length_y=1000,
                    camera_center=[150, 250],
                    H=1000,
                    W=1000,
                    img=None,
                    camera_rotation=cam_rot,
                    rotaround=0,
                    com=com.cpu().numpy(),
                    bottomview=True,
                    draw_ground_plane=False,
                    gmof_rho=contact_thresh,
                )

                # save image
                if len(imgnames) == 0:
                    filename = os.path.join(out_dir, f"{img_count:04d}.png")
                else:
                    new_img_name = "_".join(imgnames[fnum].split("/")[-6:])
                    filename = os.path.join(out_dir, new_img_name)
                print(f"Saved filename: {filename}")

                IMG = np.hstack((view1, view2, view3, img_bottom))
                IMG = pil_img.fromarray(IMG)
                # draw = ImageDraw.Draw(IMG)
                # font = ImageFont.truetype("arial.ttf", 100)
                # if in_bos_label[fnum] == 1:
                #     draw.text((0, 0), "Stable", (0, 255, 0), font=font)
                # elif in_bos_label[fnum] == -1:
                #     draw.text((0, 0), "Unstable: No contact", (255, 69, 0), font=font)
                # else:
                #     draw.text((0, 0), "Unstable: In contact", (255, 0, 0), font=font)
                IMG.save(filename)
                img_count += 1
    return in_bos_label.squeeze(), contact_metric.squeeze(), contact_mask.squeeze()


def vis_smpl_with_ground(
    thetas,
    transl,
    betas,
    seq_name,
    vis_path,
    device="cuda",
    start_idx=0,
    sub_sample=5,
    ground_offset=GROUND_OFFSETS["h36m"],
    contact_thresh=CONTACT_THRESH,
    smpl_batch_size=SMPL_BATCH_SIZE,
    visualize=True,
):
    global smpl_obj
    smpl_obj = smpl_obj.to(device)
    with torch.no_grad():
        mm2m = 1000
        if isinstance(betas, np.ndarray):
            betas = torch.FloatTensor(betas).to(device)
        if isinstance(thetas, np.ndarray):
            thetas = torch.FloatTensor(thetas).to(device)
        if isinstance(transl, np.ndarray):
            transl = torch.FloatTensor(transl).to(device)

        faces = torch.tensor(smpl_obj.faces.astype(np.int64), dtype=torch.long).to(
            device
        )

        cam_rot = trimesh.transformations.rotation_matrix(
            np.radians(-30), [1, 0, 0]
        )  # for world
        # cam_rot = trimesh.transformations.rotation_matrix(np.radians(0), [1, 0, 0]) # for cam 1
        img_count = start_idx
        out_dir = os.path.join(vis_path, seq_name)
        os.makedirs(out_dir, exist_ok=True)
        in_bos_label_accumulator = []
        contact_metric_accumulator = []
        contact_mask_accumulator = []
        for i in range(0, len(thetas), smpl_batch_size):
            gt_pose = thetas[i : i + smpl_batch_size : sub_sample, :]
            gt_beta = betas[i : i + smpl_batch_size : sub_sample, :]
            gt_transl = transl[i : i + smpl_batch_size : sub_sample, :]
            # add ground plane offset to smpl mesh along z
            gt_transl[:, -1] += -ground_offset
            smpl_output = smpl_obj(
                betas=gt_beta,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3],
                transl=gt_transl,
            )
            vertices = smpl_output.vertices
            # rotate the vertices to align with the pyrender coordinates, no aligned with y axis
            R1 = (
                ea2rm(
                    torch.tensor([[np.radians(270)]]),
                    torch.tensor([[np.radians(0)]]),
                    torch.tensor([[np.radians(0)]]),
                )
                .float()
                .to(device)
                .expand(vertices.shape[0], 3, 3)
            )  # for world
            # R1 = ea2rm(torch.tensor([[np.radians(180)]]), torch.tensor([[np.radians(0)]]),
            #            torch.tensor([[np.radians(0)]])).float().to(device).expand(vertices.shape[0], 3, 3) # for camera
            vertices = torch.einsum("bki,bji->bjk", [R1, vertices])

            # check stability (com inside bos)
            in_bos_label, com, contact_metric, contact_mask = (
                STABILITY_METRICS.check_bos_interior(
                    vertices, faces, contact_thresh=contact_thresh
                )
            )
            in_bos_label_accumulator.append(in_bos_label)
            contact_metric_accumulator.append(contact_metric)
            contact_mask_accumulator.append(contact_mask)

            for fnum in range(min(smpl_batch_size // sub_sample, len(vertices))):
                if visualize:
                    # camera values come from visual inspection in yogi/smplify-xmc-support/rendering
                    view1 = overlay_mesh(
                        vertices[fnum].cpu().numpy(),
                        smpl_obj.faces,
                        camera_transl=np.array([-1.7, 2, 7]),  # for world
                        # camera_transl=np.array([-1.2, 1.0, 9]), # for camera
                        focal_length_x=1000,
                        focal_length_y=1000,
                        camera_center=[150, 250],
                        H=1000,
                        W=1000,
                        img=None,
                        camera_rotation=cam_rot,
                        rotaround=0,
                        com=com.cpu().numpy(),
                        gmof_rho=contact_thresh,
                    )
                    # view2 = overlay_mesh(vertices[fnum].cpu().numpy(),
                    #                    smpl_obj.faces,
                    #                    camera_transl=np.array([-1.7, 2, 5]),
                    #                    focal_length_x=1000,
                    #                    focal_length_y=1000,
                    #                    camera_center=[150, 250],
                    #                    H=1000,
                    #                    W=1000,
                    #                    img=None,
                    #                    camera_rotation=cam_rot,
                    #                    rotaround=45,
                    #                    com=com.cpu().numpy(), )
                    img_bottom = overlay_mesh(
                        vertices[fnum].cpu().numpy(),
                        smpl_obj.faces,
                        camera_transl=np.array([-1.7, 2, 7]),
                        focal_length_x=1000,
                        focal_length_y=1000,
                        camera_center=[150, 250],
                        H=1000,
                        W=1000,
                        img=None,
                        camera_rotation=cam_rot,
                        rotaround=0,
                        com=com.cpu().numpy(),
                        bottomview=True,
                        draw_ground_plane=False,
                        gmof_rho=contact_thresh,
                    )

                    # save image
                    filename = os.path.join(out_dir, f"{img_count:04d}.png")
                    print(f"Saved filename: {filename}")
                    img_count += 1
                    IMG = np.hstack((view1, img_bottom))
                    IMG = pil_img.fromarray(IMG)
                    draw = ImageDraw.Draw(IMG)
                    font = ImageFont.truetype("arial.ttf", 100)
                    if in_bos_label[fnum] == 1:
                        draw.text((0, 0), "Stable", (0, 255, 0), font=font)
                    elif in_bos_label[fnum] == -1:
                        draw.text(
                            (0, 0), "Unstable: No contact", (255, 69, 0), font=font
                        )
                    else:
                        draw.text(
                            (0, 0), "Unstable: In contact", (255, 0, 0), font=font
                        )
                    IMG.save(filename)
                    # md_dict = {'beta': gt_beta[fnum],
                    #            'pose': gt_pose[fnum],
                    #            'transl': gt_transl[fnum]}
                    #
                    # # save md dict as pickle in out_dir
                    # md_name = os.path.join(out_dir, f'{img_count:04d}.pkl')
                    # with open(md_name, 'wb') as fp:
                    #     pkl.dump(md_dict, fp)
    in_bos_label_accumulator = torch.cat(in_bos_label_accumulator, dim=0)
    contact_metric_accumulator = torch.cat(contact_metric_accumulator, dim=0)
    contact_mask_accumulator = torch.cat(contact_mask_accumulator, dim=0)
    return (
        in_bos_label_accumulator.squeeze(),
        contact_metric_accumulator.squeeze(),
        contact_mask_accumulator.squeeze(),
    )


def get_view_matrix(pitch=0, yaw=0, roll=0, tx=0, ty=0, tz=0):
    camera_rot = create_rotmat(pitch, yaw, roll)

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = camera_rot
    view_matrix[0, 3] = tx
    view_matrix[1, 3] = ty
    view_matrix[2, 3] = tz

    return view_matrix


def create_rotmat(pitch, yaw, roll):
    pitch = (180 + pitch) * np.pi / 180
    yaw = yaw * np.pi / 180
    roll = roll * np.pi / 180

    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_r, sin_r = np.cos(roll), np.sin(roll)

    R = np.array(
        [
            [
                cos_y * cos_r,
                sin_p * sin_y * cos_r - cos_p * sin_r,
                cos_p * sin_y * cos_r + sin_p * sin_r,
            ],
            [
                cos_y * sin_r,
                sin_p * sin_y * sin_r + cos_p * cos_r,
                cos_p * sin_y * sin_r - sin_p * cos_r,
            ],
            [-sin_y, sin_p * cos_y, cos_p * cos_y],
        ]
    )

    return R


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
            com_pose[:3, 3] = np.array([com[0, 0], com[0, 2], 0])
        else:
            com_pose[:3, 3] = np.array(
                [com[0, 0], 0, com[0, 2]]
            )  # visualize com projection on the ground plane

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
    scene.add(com_sphere_mesh, pose=com_pose, name="com_sphere")

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
