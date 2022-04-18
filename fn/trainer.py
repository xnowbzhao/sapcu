import os
from tqdm import tqdm
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
import numpy


class Trainer():
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def cos_sim(self, x1, x2):
        scores = torch.acos(torch.cosine_similarity(x1, x2, dim=2))/numpy.pi
        return scores.mean()


    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss= self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        
        val=0.0
        num=0
        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)
            val=val+eval_step_dict.float()
            num=num+1

        return val/num
    
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device
        # Compute elbo

        points = data.get('input').to(device).float()

        output = data.get('normal').to(device).float()

        kwargs = {}

        with torch.no_grad():
            mae = self.model.compute_loss(points, output)

        return  mae

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        points = data.get('input').to(device).float()

        output = data.get('normal').to(device).float()

        c = self.model.encode_inputs(points)
        n = self.model.decode(c)
        n=n.reshape(-1)

        output=output.reshape(-1)

        loss_fn = torch.nn.MSELoss()

        loss = loss_fn(n, output)
    
        return loss.float()
