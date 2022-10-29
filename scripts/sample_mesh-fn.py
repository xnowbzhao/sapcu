import argparse
import trimesh
import numpy as np
import os
import glob
import sys
import time

from multiprocessing import Pool
from functools import partial
# TODO: do this better
sys.path.append('..')
from im2mesh.utils import binvox_rw, voxels
from im2mesh.utils.libmesh import check_mesh_contains
import sklearn.neighbors


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
parser.add_argument('--pointcloud_size', type=int, default=800000,
                    help='Size of point cloud.')

parser.add_argument('--voxels_folder', type=str,
                    help='Output path for voxelization.')
parser.add_argument('--voxels_res', type=int, default=32,
                    help='Resolution for voxelization.')

parser.add_argument('--pointing_folder', type=str,
                    help='Output path for points.')
parser.add_argument('--pointing_size', type=int, default=50000,
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

parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')
    
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
   
    if args.pointing_folder is not None:
        export_pointing(mesh, modelname, loc, scale, args)


def export_pointing(mesh, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return
    

    points, face_idx = trimesh.sample.sample_surface_even(mesh, args.pointcloud_size)
    normals = mesh.face_normals[face_idx]

    dtype = np.float32

    points = points.astype(dtype)
    normals = normals.astype(dtype)

    
    boxlength=2.0
    step1=boxlength/40
    step2=step1/10

    tree=sklearn.neighbors.KDTree(points)
    
    #divide the space into voxels and use the center of each voxel as output points 
    
    mt1=np.indices((50,50,50)).transpose(1,2,3,0)
    xyz1=mt1*step1-1.0
    xyz1=xyz1.reshape((50*50*50,3))
    center1=xyz1+step1/2
    dist, idx1 = tree.query(center1, 1)
    dist=dist.squeeze(1)
    xyz1=xyz1[np.where(dist<step1+0.01),:].squeeze(0)

    mt2=np.indices((10,10,10)).transpose(1,2,3,0)
    mt2=np.expand_dims(mt2.reshape(1000,3),0)
    mt2=np.tile(mt2,(xyz1.shape[0],1,1))
    mt2=mt2*step2
    xyz2=np.expand_dims(xyz1,1)
    xyz2=np.tile(xyz2,(1,mt2.shape[1],1))+mt2
    center2=xyz2
    center2=center2.reshape((center2.shape[0]*center2.shape[1],3))
    center2=center2+step2/2
    
    #add noise
    
    noisy=np.random.rand(center2.shape[0],center2.shape[1])*0.001
    center2=center2+noisy

    #sample 10 points 
    tinput= center2
    dist, idx1 = tree.query(tinput, 10)
    dist=dist[:,0]

    #select points in [0.003, 0.03]
    tindex=np.where((dist>=0.003)& (dist<=0.03))[0]
    np.random.shuffle(tindex)
    
    if tindex.shape[0]>args.pointing_size:
        tindex=tindex[0:args.pointing_size]
    tinput=tinput[tindex,:]
    idx1=idx1[tindex,:]
    
    # compute the average vector   
    for i in range(10):
        if i == 0:
            output=points[idx1[:,i],:]
        else:
            output=output+points[idx1[:,i],:]

 
    output=output/10

    normal=(output-tinput)
    length= np.tile(np.expand_dims(np.sqrt(np.sum(normal*normal,1)),1),(1,3))
    normal=normal/length
    
    points = points.astype(dtype)
    filename = os.path.join(args.pointing_folder, modelname + '.npz')
    print('Writing pointing: %s' % filename)
    np.savez(filename, points=tinput, pointing=normal)

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)