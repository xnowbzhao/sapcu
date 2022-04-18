import torch
import torch.optim as optim
from torch import autograd
import numpy as np
import math
from tqdm import trange
import trimesh
import copy
import time
from tqdm import tqdm
from sklearn.neighbors import KDTree
import torch.nn.functional as F
import os
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


class Generator3D6(object):

    def __init__(self, model1, model2, device):
        self.model1 = model1
        self.model2 = model2
        self.device = device

    def upsample(self, data):

        pc = self.generateiopoint(data)

        return pc

    def generateiopoint(self, data):

        data=np.squeeze(data,0)
        tree1=KDTree(data)
        
        print("generate seedpoint")
        wq="./dense"
        wq=wq+" 0.004 "+str(data.shape[0])
        
        print(wq)
        os.system(wq)
        data2=np.loadtxt("target.xyz")
        xyz2=data2[:,0:3]
        
        print("fn")        
        pp=xyz2.shape[0]//400
        p_split = np.array_split(xyz2, pp, axis=0)
        normal=None
        for i in tqdm(range(len(p_split))):

            dist,  idx = tree1.query(p_split[i], 100)
            cloud=data[idx]
            cloud=cloud-np.tile(np.expand_dims(p_split[i],1),(1,100,1))
            with torch.no_grad():
                c = self.model1.encode_inputs(torch.from_numpy(np.expand_dims(cloud,0)).float().to(self.device))
            with torch.no_grad():
                n = self.model1.decode(c)
            n=n.detach().cpu().numpy()
            if normal is None:
                normal=n
            else:
                normal= np.append(normal,n,axis=0)


        n_split = np.array_split(normal, pp, axis=0)
        xyzout=[]
        
        print("fd") 
        for i in tqdm(range(len(p_split))):

            dist,  idx = tree1.query(p_split[i], 100)
            cloud=data[idx]
            cloud=cloud-np.tile(np.expand_dims(p_split[i],1),(1,100,1))

            for j in range(cloud.shape[0]):
                M1=rotation_matrix_from_vectors(n_split[i][j],[1,0,0])
                cloud[j]=(np.matmul(M1,cloud[j].T)).T

            with torch.no_grad():
                c = self.model2.encode_inputs(torch.from_numpy(np.expand_dims(cloud,0)).float().to(self.device))
            with torch.no_grad():
                n = self.model2.decode(c)

            length=np.tile(np.expand_dims(n.detach().cpu().numpy(),1),(1,3))
            xyzout.extend((p_split[i]+n_split[i]*length).tolist())

        xyzout=np.array(xyzout)
        
        print("remove outliers")  
        tree3=KDTree(xyzout)
        dist, idx = tree3.query(xyzout, 30)
        avg=np.mean(dist,axis=1)
        avgtotal=np.mean(dist)
        idx=np.where(avg<avgtotal*1.5)[0]
        xyzout=xyzout[idx,:]


        return xyzout