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
from ply import read_ply

# params
VOXEL_SIZE = 0.05  # in meters (?)
N_FEATURES = 16  # if this is changed, then will need to change model weights file

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
model = ResUNetBN2C(1, N_FEATURES, normalize_feature=True, conv1_kernel_size=3, D=3)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model = model.to(device)
print('Loaded model ' + model_path)

# load point clouds
def load_cloud(path):
    print('Loading point cloud at ' + path)
    data = read_ply(path)
    classes = data['class']
    points = np.vstack((data['x'], data['y'], data['z'])).T
    # pcd = np.array(o3d.io.read_point_cloud(path).points)
    print('\tshape: ', points.shape)
    return points, classes

all_features = []
all_labels = []

for name in ['MiniLille1', 'MiniLille2', 'MiniParis1']:
    pts, lbs = load_cloud(f'dataset/training/{name}.ply')
    k = 150000  # batch size 

    assert(len(pts) == len(lbs))

    for i in range(len(pts)//k+1):
        points = pts[i*k:min((i+1)*k,len(pts))]
        labels = lbs[i*k:min((i+1)*k,len(pts))]
    
        feats = []
        feats.append(np.ones((len(points), 1)))

        feats = np.hstack(feats)

        # Voxelize points and feats
        coords = np.floor(points / VOXEL_SIZE)
        inds = ME.utils.sparse_quantize(coords, return_index=True)

        # build map voxel xyz -> ind
        # then for pt xyz, retrieve its features with
        #       features[map[np.floor(xyz/VOXEL_SIZE)]]
        voxel2id = {tuple(map(int, coords[ind])): ind for ind in inds}

        coords = coords[inds]
        # Convert to batched coords compatible with ME
        coords = ME.utils.batched_coordinates([coords])
        return_coords = points[inds]

        feats = feats[inds]

        feats = torch.tensor(feats, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.int32)

        stensor = ME.SparseTensor(feats, coords=coords).to(device)

        xyz_down, features = return_coords, model(stensor).F
        print('\tfeatures: ', features.shape)

        labels = labels[inds]

        all_features.append(features)
        all_labels.append(labels)
