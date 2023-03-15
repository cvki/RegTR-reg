import sys
sys.path.append('/d2/code/gitRes/Expts/RegTR-main')

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from src.cvhelpers.lie.numpy import SE3

import os.path
import time
import numpy as np
from tqdm import tqdm

# a=np.array([i for i in range(21)]).reshape(3,7)
# print(a)
# b=[np.sum(a<a.shape[1],axis=1)]
# print(b)

import open3d as o3d
import torch

def visualizepc1(pts,str='pcd'):   # np--pts
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts)
    pcd1.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([pcd1],window_name=str)

'pts should be cpu valiables'
def visualizepc2(pts1,pts2, str='pcd1,pcd2'):  # np--pts
    if isinstance(pts1,torch.Tensor):
        pts1=pts1.numpy()
    if isinstance(pts2,torch.Tensor):
        pts2=pts2.numpy()
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd1.paint_uniform_color([1,0,0])   # red
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    pcd2.paint_uniform_color([0,1,0])   # green
    o3d.visualization.draw_geometries([pcd1,pcd2], window_name=str)

# def visualizepc3(*pts, str='pcd1-3'):  # np--pts
def visualizepc3(pts1,pts2,pts3, str='pcd1-3'):  # np--pts
    # pts1,pts2,pts3=pts
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd1.paint_uniform_color([1,0,0])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    pcd2.paint_uniform_color([0,1,0])
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(pts3)
    pcd3.paint_uniform_color([0,0,1])
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3], window_name=str)



def _sample_pose_small(std=0.1):
        perturb = SE3.sample_small(std=std).as_matrix()
        return torch.from_numpy(perturb).float()
    
def _sample_pose_large():
        euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
        rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
        perturb = np.concatenate([rot_ab, np.zeros((3, 1))], axis=1)
        return torch.from_numpy(perturb).float()
    
    
large_mat=_sample_pose_large()
small_mat=_sample_pose_small()
pts1=torch.rand(size=(1024,3))



print(f'large_mat is:\n{large_mat}\n small_mat is:\n {small_mat}')
