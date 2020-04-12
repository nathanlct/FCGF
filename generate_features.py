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


class FCGF_Features(object):
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size  # in meters
        self.n_features = 16  # do not change, model is pre-trained on 16 features
        self.max_batch_size = 200000  # depends on GPU mem available

        self._load_network()

    def _load_network(self):
        # use GPU if available
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using ' + device_str)
        self.device = torch.device(device_str)

        # load model
        model_path = 'ResUNetBN2C-16feat-3conv.pth'
        if not os.path.isfile(model_path):
            print('Downloading weights...')
            urlretrieve(
                "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
                'ResUNetBN2C-16feat-3conv.pth')
        checkpoint = torch.load(model_path)
        model = ResUNetBN2C(1, self.n_features, normalize_feature=True, conv1_kernel_size=3, D=3)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model = model.to(self.device)
        print('Loaded model ' + model_path)

    def _load_cloud(self, path, load_labels=True):
        print('Loading point cloud at ' + path)
        data = read_ply(path)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        print('\tshape: ', points.shape)
        labels = data['class'] if load_labels else np.zeros((points.shape[0],))
        return points, labels

    def generate_features(self, root_path, ply_name, generate_labels=True):
        points, labels = self._load_cloud(f'{root_path}{ply_name}.ply', load_labels=generate_labels)
        assert(len(points) == len(labels))

        # sort points by increasing x
        points, labels = np.array(points), np.array(labels)
        idx = np.argsort(points, axis=0)[:,0]
        # points, labels = list(zip(*sorted(zip(points, labels), key=lambda x: x[0][0])))
        #print(idx[:100])
        points, labels = points[idx], labels[idx]
        #points, labels = points[len(points)//2:], labels[len(labels)//2:]

        all_features = np.zeros((len(points), self.n_features))

        # for each batch of points
        for i in range(len(points) // self.max_batch_size + 1):
            batch_start = i * self.max_batch_size
            batch_end = min((i + 1) * self.max_batch_size, len(points))
            points_batch = points[batch_start:batch_end]

            #print('points batch', points_batch.shape)
        
            feats = []
            feats.append(np.ones((len(points_batch), 1)))
            feats = np.hstack(feats)

            # voxelize points and feats
            coords = np.floor(points_batch / self.voxel_size)
            inds = ME.utils.sparse_quantize(coords, return_index=True)

            #print('inds', np.array(inds).shape)

            # convert to batched coords compatible with ME
            coords = coords[inds]
            feats = feats[inds]

            coords = ME.utils.batched_coordinates([coords])
            return_coords = points_batch[inds]  # useless?

            feats = torch.tensor(feats, dtype=torch.float32)
            coords = torch.tensor(coords, dtype=torch.int32)

            stensor = ME.SparseTensor(feats, coords=coords).to(self.device)

            # generate features for voxels
            xyz_down, features = return_coords, self.model(stensor).F
            features = features.cpu().detach().numpy()
            print('\tfeatures: ', features.shape)
        
            # build map voxel xyz -> features
            voxel2feat = {tuple(map(int, coords[j][1:])): features[j] for j in range(len(features))}

            #print(coords.shape, features.shape)
            #print(len(list(voxel2feat.keys())))
            #print(list(voxel2feat.keys())[0])

            # deduce features for all points
            for j in range(batch_start, batch_end):
                pt = points[j]
                pt_voxel = tuple(map(int, np.floor(pt / self.voxel_size)))

                all_features[j,:] = voxel2feat[pt_voxel]

        # save labels and features for all points
        np.save(f'{ply_name}_features', all_features)
        if generate_labels:
            np.save(f'{ply_name}_labels', labels)

    def generate_features_reduced(self, root_path, ply_name, generate_labels=True):
        points, labels = self._load_cloud(f'{root_path}{ply_name}.ply', load_labels=generate_labels)
        assert(len(points) == len(labels))

        # sort points by increasing x
        points, labels = np.array(points), np.array(labels)
        idx = np.argsort(points, axis=0)[:,0]
        # points, labels = list(zip(*sorted(zip(points, labels), key=lambda x: x[0][0])))
        #print(idx[:100])
        points, labels = points[idx], labels[idx]
        #points, labels = points[len(points)//2:], labels[len(labels)//2:]

        all_features = []
        all_labels = []

        # for each batch of points
        for i in range(len(points) // self.max_batch_size + 1):
            batch_start = i * self.max_batch_size
            batch_end = min((i + 1) * self.max_batch_size, len(points))
            points_batch = points[batch_start:batch_end]
            labels_batch = labels[batch_start:batch_end]

            #print('points batch', points_batch.shape)
        
            feats = []
            feats.append(np.ones((len(points_batch), 1)))
            feats = np.hstack(feats)

            # voxelize points and feats
            coords = np.floor(points_batch / self.voxel_size)
            inds = ME.utils.sparse_quantize(coords, return_index=True)

            #print('inds', np.array(inds).shape)

            # convert to batched coords compatible with ME
            coords = coords[inds]
            feats = feats[inds]

            coords = ME.utils.batched_coordinates([coords])
            return_coords = points_batch[inds]  # useless?

            feats = torch.tensor(feats, dtype=torch.float32)
            coords = torch.tensor(coords, dtype=torch.int32)

            stensor = ME.SparseTensor(feats, coords=coords).to(self.device)

            # generate features for voxels
            xyz_down, features = return_coords, self.model(stensor).F
            features = features.cpu().detach().numpy()
            print('\tfeatures: ', features.shape)
        
            all_features.append(features)
            all_labels.append(labels_batch[inds])

        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)

        print(all_features.shape, all_labels.shape)

        # save labels and features for all points
        np.save(f'{ply_name}_features_reduced', all_features)
        if generate_labels:
            np.save(f'{ply_name}_labels_reduced', all_labels)


if __name__ == '__main__':
    network = FCGF_Features(voxel_size=0.05)#1)

    # network.generate_features('dataset/', 'Lille1_1')
    # network.generate_features('dataset/', 'Lille1_2')
    # network.generate_features('dataset/', 'Lille2')
    # network.generate_features('dataset/', 'Paris')
    
    network.generate_features_reduced('dataset_small/training/', 'MiniLille2')
    network.generate_features_reduced('dataset_small/training/', 'MiniParis1')
    network.generate_features_reduced('dataset_small/training/', 'MiniLille1')
    #network.generate_features('dataset/test/', 'MiniDijon9', generate_labels=False)


# for step, files in [('training', ['MiniLille1', 'MiniLille2', 'MiniParis1']), ('test', ['MiniDijon9'])]:
#     print(f'\nGenerating {step} features')

#     all_features = []
#     all_labels = []

#     for name in files:
#         pts, lbs = load_cloud(f'dataset/{step}/{name}.ply')
#         k = 150000  # batch size 

#         assert(len(pts) == len(lbs))

#         # sort points by increasing x
#         pts, lbs = list(zip(*sorted(zip(pts, lbs), key=lambda x: x[0][0])))
#         pts = np.array(pts)
#         lbs = np.array(lbs)

#         total_feats = 0

#         for i in range(len(pts)//k+1):
#             points = pts[i*k:min((i+1)*k,len(pts))]
#             labels = lbs[i*k:min((i+1)*k,len(pts))]
        
#             feats = []
#             feats.append(np.ones((len(points), 1)))

#             feats = np.hstack(feats)

#             # Voxelize points and feats
#             coords = np.floor(points / VOXEL_SIZE)
#             inds = ME.utils.sparse_quantize(coords, return_index=True)

#             # build map voxel xyz -> ind
#             # then for pt xyz, retrieve its features with
#             #       features[map[np.floor(xyz/VOXEL_SIZE)]]
#             voxel2id = {tuple(map(int, coords[ind])): ind for ind in inds}

#             coords = coords[inds]
#             # Convert to batched coords compatible with ME
#             coords = ME.utils.batched_coordinates([coords])
#             return_coords = points[inds]

#             feats = feats[inds]

#             feats = torch.tensor(feats, dtype=torch.float32)
#             coords = torch.tensor(coords, dtype=torch.int32)

#             stensor = ME.SparseTensor(feats, coords=coords).to(device)

#             xyz_down, features = return_coords, model(stensor).F
#             print('\tfeatures: ', features.shape)
#             total_feats += features.shape[0]

#             labels = labels[inds]

#             all_features.append(features.cpu().detach().numpy())
#             all_labels.append(np.array(labels).reshape(-1, 1))
#         print('features: ', total_feats)


#     all_features = np.vstack(tuple(all_features))
#     all_labels = np.vstack(tuple(all_labels))

#     print('All features: ', all_features.shape)
#     print('All labels: ', all_labels.shape)

#     print('Saving...')

#     np.save(f'{step}_features', all_features)
#     np.save(f'{step}_labels', all_labels)