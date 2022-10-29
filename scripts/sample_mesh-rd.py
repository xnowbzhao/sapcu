import argparse
import trimesh
import numpy as np
import os
import glob
import sys
import time
import random
import math

from multiprocessing import Pool
from functools import partial
# TODO: do this better
sys.path.append('..')

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')

parser.add_argument('--rotate_xz', type=float, default=0.,
                    help='Angle to rotate around y axis.')

parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--bbox_in_folder', type=str,
                    help='Path to other input folder to extract'
                         'bounding boxes.')

parser.add_argument('--pointcloud_folder', type=str,
                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=50000,
                    help='Size of point cloud.')

parser.add_argument('--voxels_folder', type=str,
                    help='Output path for voxelization.')
parser.add_argument('--voxels_res', type=int, default=32,
                    help='Resolution for voxelization.')

parser.add_argument('--points_folder', type=str,
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=50000,
                    help='Size of points.')
parser.add_argument('--ray_size', type=int, default=100000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=1.,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.2,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')
parser.add_argument('--filtered_folder', type=str,
                    help='filter mesh.')
parser.add_argument('--ray_folder', type=str,
                    help='ray.')
parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')

def randnsphere(n):
    v = [random.gauss(0, 1) for i in range(0, n)]
    inv_len = 1.0 / math.sqrt(sum(coord * coord for coord in v))
    return [coord * inv_len for coord in v]

def main(args):
    input_files = glob.glob(os.path.join(args.in_folder, '*.off'))
    np.random.seed(0)
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args)


def process_path(in_path, args):
    in_file = os.path.basename(in_path)
    modelname = os.path.splitext(in_file)[0]
    mesh = trimesh.load(in_path, process=False)
    loc = np.zeros(3)
    scale = 1.
   
    ray(mesh, modelname, loc, scale, args)


def ray(mesh, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return
    
    filename = os.path.join(args.ray_folder,
                            modelname + '.npz')
    
    Intersector=trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    #Fast implement: do not generate seed points and project them to the surface 
    #but select the intersections on the surface and randomly generate length and directions
    
    
    #select the intersection of ray and surface
    outputs, face_idx = mesh.sample(args.ray_size, return_index=True)
    
    #randomly generate normal and length to obtain seed points, and use the reversed normals as directions
    normals=np.array( [randnsphere(3) for i in range(0, args.ray_size)])
    lens= np.random.rand(args.ray_size,3)*0.027+0.003
    lens[:, 2]=  lens[:, 1]=  lens[:, 0]
    points=outputs+lens*normals
    normals=normals*-1
    
    #remove the seed points that intersects with other faces
    firstindex=Intersector.intersects_first(points,normals)
    cc=np.where(face_idx == firstindex)[0]
      
    face_idx = face_idx[cc]
    points=points[cc,:]
    normals=normals[cc,:]
    lens=lens[cc]
    
    #remove the seed points with too large intersection angle
    facenormal = mesh.face_normals[face_idx]
    dot=np.sum(facenormal*(-normals), axis=1)

    angle = np.arccos(dot)
    cc=np.where(angle<1)[0]
    
    points=points[cc,:]
    normals=normals[cc,:]
    lens=lens[cc]   
    
    
    dtype = np.float32

    points = points.astype(dtype)
    outputs = outputs.astype(dtype)
    lens = lens.astype(dtype)

    print('Writing points: %s' % filename)
    np.savez(filename, points=points, normals=normals, lens=lens)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
