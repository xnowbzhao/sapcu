import yaml
from fd import field
import torch
from torchvision import transforms
from fd import datacore
from fd import coder
# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

# Datasets

def get_dataset(mode, cfg):
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }
    split = splits[mode]
    dataset_folder = cfg['data']['path']
    

    fields =[ field.PointCloudField(cfg['data']['pointcloud_file'])
    , field.fdField()]
    
    dataset = datacore.Shapes3dDataset(dataset_folder, fields, split=split)
    return dataset

def get_model(cfg, device):

    dim = cfg['model']['dim']
    c_dim = cfg['model']['c_dim']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    decoder = coder.pyramid_Decoder3()
    encoder = coder.DGCNN_cls(20, 1024)
    #encoder=coder.ResnetPointnet()
    model = coder.OccupancyNetwork(decoder, encoder, device=device)


    return model


