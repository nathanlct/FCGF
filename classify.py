import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from util.visualization import get_colored_point_cloud_feature
from util.misc import extract_features
import torch
import MinkowskiEngine as ME
from model.resunet import ResUNetBN2C

# params
VOXEL_SIZE = 0.025  # in meters (?)

# use GPU if available
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ' + device_str)
device = torch.device(device_str)

# load model
model_path = 'ResUNetBN2C-16feat-3conv.pth'
if not os.path.isfile(model_path):
  print('Downloading weights...')
  urlretrieve(
      "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
      'ResUNetBN2C-16feat-3conv.pth')
checkpoint = torch.load(model_path)
model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model = model.to(device)
print('Loaded model ' + model_path)

# load point clouds
def load_cloud(path):
    print('Loading point cloud at ' + path)
    pcd = np.array(o3d.io.read_point_cloud(path).points)
    print('\tshape: ', pcd.shape)

pts1 = load_cloud('dataset/training/MiniLille1.ply')
pts2 = load_cloud('dataset/training/MiniLille2.ply')
pts3 = load_cloud('dataset/training/MiniParis1.ply')
