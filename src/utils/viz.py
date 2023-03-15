import numpy as np
import torch
from matplotlib.pyplot import cm as colormap

import src.cvhelpers.visualization as cvv
import src.cvhelpers.colors as colors
from src.cvhelpers.torch_helpers import to_numpy
from src.utils.se3_torch import se3_transform

import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.utils.open3d import (
    make_open3d_point_cloud,
    make_open3d_axes,
    make_open3d_corr_lines,
)


def visualize_registration(src_xyz, tgt_xyz, correspondences,
                           correspondence_conf=None,
                           pose_gt=None, pose_pred=None):
    """Visualize registration, shown as a 2x3 grid:

    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------

    0: Source point cloud with source keypoints
    1: Source and target point clouds, with lines indicating source keypoints to
       their transformed locations
    2: Source and target point clouds under groundtruth alignment (without clutter)
    3: Target point cloud with predicted transformed source keypoints
    4: Source and target point clouds under groundtruth alignment, with
       source keypoints and predited transformed coordinates, and a lines joining
       them (shorter lines means more accurate predictions)
    5: Source and target point clouds under predicted alignment (without clutter)

    Created 22 Oct 2021
    """

    if pose_gt is None:
        src_xyz_warped = src_xyz
        src_corr_warped = correspondences[:, :3]
    else:
        src_xyz_warped = se3_transform(pose_gt, src_xyz)
        src_corr_warped = se3_transform(pose_gt, correspondences[:, :3])

    vis = cvv.Visualizer(num_renderers=6, win_size=(1850, 1200))

    if correspondence_conf is None:
        src_kp_color = (255, 128, 128)
        tgt_kp_color = (128, 255, 128)
    else:
        conf = to_numpy(correspondence_conf)
        src_color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('autumn'))
        src_kp_color = (src_color_mapper.to_rgba(conf)[:, :3] * 255).astype(np.uint8)
        tgt_color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('summer'))
        tgt_kp_color = (tgt_color_mapper.to_rgba(conf)[:, :3] * 255).astype(np.uint8)
    # Show points on source
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=0,
    )
    vis.add_object(
        cvv.create_point_cloud(src_corr_warped, colors=src_kp_color, pt_size=4),
        renderer_idx=0,
    )

    # Show points on target
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=3,
    )
    vis.add_object(
        cvv.create_point_cloud(correspondences[:, 3:], colors=tgt_kp_color, pt_size=4),
        renderer_idx=3,
    )

    # Show correspondences with lines joining the two
    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=1,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=1,
    )
    vis.add_object(
        cvv.create_lines(correspondences),
        renderer_idx=1
    )

    # Show overlap using groundtruth pose
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=4,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=4,
    )
    vis.add_object(
        cvv.create_point_cloud(src_corr_warped, colors=src_kp_color, pt_size=4),
        renderer_idx=4
    )
    vis.add_object(
        cvv.create_point_cloud(correspondences[:, 3:], colors=tgt_kp_color, pt_size=4),
        renderer_idx=4
    )
    vis.add_object(
        cvv.create_lines(torch.cat([src_corr_warped, correspondences[:, 3:]], dim=1)),
        renderer_idx=4
    )

    # Show groundtruth (without clutter)
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=2,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=2,
    )

    # Show predicted pose
    if pose_pred is not None:
        vis.add_object(
            cvv.create_point_cloud(se3_transform(pose_pred, src_xyz), colors=colors.RED),
            renderer_idx=5,
        )
        vis.add_object(
            cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
            renderer_idx=5,
        )

    # Render loop
    vis.reset_camera()
    vis.start()



def visualizepc1(pts, str_tittle='pcd'):    # np--pts
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts)
    pcd1.paint_uniform_color([1, 0, 0])     # red
    o3d.visualization.draw_geometries([pcd1], window_name=str_tittle)


'pts should be cpu valiables'


def visualizepc2(pts1, pts2, str_tittle='pcd1,pcd2'):  # np--pts
    if isinstance(pts1, torch.Tensor):
        pts1 = pts1.numpy()
    if isinstance(pts2, torch.Tensor):
        pts2 = pts2.numpy()
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd1.paint_uniform_color([1, 0, 0])  # red
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    pcd2.paint_uniform_color([0, 1, 0])  # green
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name=str_tittle)


# def visualizepc3(*pts, str='pcd1-3'):  # np--pts
def visualizepc3(pts1, pts2, pts3, str_tittle='pcd1-3'):  # np--pts
    # pts1,pts2,pts3=pts
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd1.paint_uniform_color([1, 0, 0])     # red
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    pcd2.paint_uniform_color([0, 1, 0])     # green
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(pts3)
    pcd3.paint_uniform_color([0, 0, 1])     # blue
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3], window_name=str_tittle)


def draw_point_to_node(points, nodes, point_to_node, node_colors=None):
    if node_colors is None:
        node_colors = np.random.rand(*nodes.shape)
    # point_colors = node_colors[point_to_node] * make_scaling_along_axis(points, alpha=0.3).reshape(-1, 1)
    point_colors = node_colors[point_to_node]
    node_colors = np.ones_like(nodes) * np.array([[1, 0, 0]])

    ncd = make_open3d_point_cloud(nodes, colors=node_colors)
    pcd = make_open3d_point_cloud(points, colors=point_colors)
    axes = make_open3d_axes()

    o3d.visualization.draw([pcd, ncd, axes])


def draw_node_correspondences(
        ref_points,
        ref_nodes,
        ref_point_to_node,
        src_points,
        src_nodes,
        src_point_to_node,
        node_correspondences,
        ref_node_colors=None,
        src_node_colors=None,
        offsets=(0, 2, 0),
):
    src_nodes = src_nodes + offsets
    src_points = src_points + offsets

    if ref_node_colors is None:
        ref_node_colors = np.random.rand(*ref_nodes.shape)
    # src_point_colors = src_node_colors[src_point_to_node] * make_scaling_along_axis(src_points).reshape(-1, 1)
    ref_point_colors = ref_node_colors[ref_point_to_node]
    ref_node_colors = np.ones_like(ref_nodes) * np.array([[1, 0, 0]])

    if src_node_colors is None:
        src_node_colors = np.random.rand(*src_nodes.shape)
    # tgt_point_colors = tgt_node_colors[tgt_point_to_node] * make_scaling_along_axis(tgt_points).reshape(-1, 1)
    src_point_colors = src_node_colors[src_point_to_node]
    src_node_colors = np.ones_like(src_nodes) * np.array([[1, 0, 0]])

    ref_ncd = make_open3d_point_cloud(ref_nodes, colors=ref_node_colors)
    ref_pcd = make_open3d_point_cloud(ref_points, colors=ref_point_colors)
    src_ncd = make_open3d_point_cloud(src_nodes, colors=src_node_colors)
    src_pcd = make_open3d_point_cloud(src_points, colors=src_point_colors)
    corr_lines = make_open3d_corr_lines(ref_nodes, src_nodes, node_correspondences)
    axes = make_open3d_axes(scale=0.1)

    o3d.visualization.draw([ref_pcd, ref_ncd, src_pcd, src_ncd, corr_lines, axes])


def get_colors_with_tsne(data):
    r"""
    Use t-SNE to project high-dimension feats to rgbd
    :param data: (N, C)
    :return colors: (N, 3)
    """
    tsne = TSNE(n_components=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data).reshape(-1)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    normalized_tsne_results = (tsne_results - tsne_min) / (tsne_max - tsne_min)
    colors = plt.cm.Spectral(normalized_tsne_results)[:, :3]
    return colors


def write_points_to_obj(file_name, points, colors=None, radius=0.02, resolution=6):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    vertices = np.asarray(sphere.vertices)
    triangles = np.asarray(sphere.triangles) + 1

    v_lines = []
    f_lines = []

    num_point = points.shape[0]
    for i in tqdm(range(num_point)):
        n = i * vertices.shape[0]

        for j in range(vertices.shape[0]):
            new_vertex = points[i] + vertices[j]
            line = 'v {:.6f} {:.6f} {:.6f}'.format(new_vertex[0], new_vertex[1], new_vertex[2])
            if colors is not None:
                line += ' {:.6f} {:.6f} {:.6f}'.format(colors[i, 0], colors[i, 1], colors[i, 2])
            v_lines.append(line + '\n')

        for j in range(triangles.shape[0]):
            new_triangle = triangles[j] + n
            line = 'f {} {} {}\n'.format(new_triangle[0], new_triangle[1], new_triangle[2])
            f_lines.append(line)

    with open(file_name, 'w') as f:
        f.writelines(v_lines)
        f.writelines(f_lines)


def convert_points_to_mesh(points, colors=None, radius=0.02, resolution=6):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    vertices = np.asarray(sphere.vertices)
    triangles = np.asarray(sphere.triangles)

    new_vertices = points[:, None, :] + vertices[None, :, :]
    if colors is not None:
        new_vertex_colors = np.broadcast_to(colors[:, None, :], new_vertices.shape)
    new_vertices = new_vertices.reshape(-1, 3)
    new_vertex_colors = new_vertex_colors.reshape(-1, 3)
    bases = np.arange(points.shape[0]) * vertices.shape[0]
    new_triangles = bases[:, None, None] + triangles[None, :, :]
    new_triangles = new_triangles.reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(new_vertex_colors)
    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)

    return mesh


def write_points_to_ply(file_name, points, colors=None, radius=0.02, resolution=6):
    mesh = convert_points_to_mesh(points, colors=colors, radius=radius, resolution=resolution)
    o3d.io.write_triangle_mesh(file_name, mesh, write_vertex_normals=False)


def write_correspondences_to_obj(file_name, src_corr_points, tgt_corr_points):
    v_lines = []
    l_lines = []

    num_corr = src_corr_points.shape[0]
    for i in tqdm(range(num_corr)):
        n = i * 2

        src_point = src_corr_points[i]
        tgt_point = tgt_corr_points[i]

        line = 'v {:.6f} {:.6f} {:.6f}\n'.format(src_point[0], src_point[1], src_point[2])
        v_lines.append(line)

        line = 'v {:.6f} {:.6f} {:.6f}\n'.format(tgt_point[0], tgt_point[1], tgt_point[2])
        v_lines.append(line)

        line = 'l {} {}\n'.format(n + 1, n + 2)
        l_lines.append(line)

    with open(file_name, 'w') as f:
        f.writelines(v_lines)
        f.writelines(l_lines)

