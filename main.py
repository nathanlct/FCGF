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

import torch

if not os.path.isfile('ResUNetBN2C-16feat-3conv.pth'):
  print('Downloading weights...')
  urlretrieve(
      "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
      'ResUNetBN2C-16feat-3conv.pth')

if not os.path.isfile('redkitchen-20.ply'):
  print('Downloading a mesh...')
  urlretrieve("https://node1.chrischoy.org/data/publications/fcgf/redkitchen-20.ply",
              'redkitchen-20.ply')


# use GPU if available
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ' + device_str)
device = torch.device(device_str)

# load model
model_path = 'ResUNetBN2C-16feat-3conv.pth'
checkpoint = torch.load(model_path)
model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model = model.to(device)
print('Loaded model ' + model_path)


voxel_size = 0.025

# load point cloud
input_path = 'redkitchen-20.ply'
pcd = o3d.io.read_point_cloud(input_path)
points = np.array(pcd.points)
print('Loaded point cloud ' + input_path)

# tmp test
#points = points[:10]

print('initial points: ', points.shape)

#print(points)

#print('pre-voxelized:')
#print(np.floor(points / voxel_size))

# extract points
print('Extract features')

model.eval()

feats = []
feats.append(np.ones((len(points), 1)))

feats = np.hstack(feats)

# Voxelize points and feats
coords = np.floor(points / voxel_size)
inds = ME.utils.sparse_quantize(coords, return_index=True)
coords = coords[inds]
# Convert to batched coords compatible with ME
coords = ME.utils.batched_coordinates([coords])
return_coords = points[inds]

feats = feats[inds]

feats = torch.tensor(feats, dtype=torch.float32)
coords = torch.tensor(coords, dtype=torch.int32)

stensor = ME.SparseTensor(feats, coords=coords).to(device)

xyz_down, feature = return_coords, model(stensor).F


print('points after voxelization: ', xyz_down.shape)
#print(np.floor(xyz_down / voxel_size))

print('returned features: ', feature.shape)