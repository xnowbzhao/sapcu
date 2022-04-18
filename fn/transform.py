import numpy as np
from sklearn.neighbors import KDTree
import math

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

def PointcloudNoise(data, stddev):
    if stddev == 0:
        return data
    
    data_out = data.copy()
    points = data[None]
    
    noise = stddev * np.random.randn(*points.shape)
    
    noise = noise.astype(np.float32)
    
    data_out[None] = points + noise
    
    return data_out
	

def SubsamplePointcloud(data, N):
    data_out = data.copy()
    points = data['cloud']
    indices = np.random.randint(points.shape[0], size=N)
    data_out['cloud'] = points[indices, :]

    return data_out



def Subsamplefn(data,N, M):
    data_out = data.copy()
    
    indices = np.random.randint(data['input'].shape[0], size=N) 
    tinput= data_out['input'][indices,:]
    data_out['normal']= data_out['normal'][indices,:]
       
    tree=KDTree(data_out['cloud'])
    
    
    dist,  idx = tree.query(tinput, M)
    data_out['input']=data_out['cloud'][idx]
    tinput=np.tile(np.expand_dims(tinput,1),(1,M,1))
    data_out['input']= data_out['input']-tinput

    return data_out


def GdataKNN(data):
    data=SubsamplePointcloud(data,2048)
    data=Subsamplefn(data, 16, 100)
    
    return data
