import os
import glob
import random
import numpy as np
import trimesh
from fn import transform as tsf

    
class PointCloudField():
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_name):
        self.file_name = file_name


    def load(self, model_path):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)
        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)

        data = {
            'cloud': points,
        }

        return data


    
class fnField():

    def __init__(self):
        self.file_name = 'fn.npz'
    def load(self, model_path):

        file_path = os.path.join(model_path, self.file_name)
        fn_dict = np.load(file_path)
        points = fn_dict['points'].astype(np.float32)
        normals = fn_dict['normals'].astype(np.float32)
        data = {
            'input': points,
            'normal': normals,
        }

        return data

