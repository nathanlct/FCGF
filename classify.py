import numpy as np


names = ['Lille1_1', 'Lille1_2', 'Lille2', 'Paris1', 'Paris2']

features = np.load(f'{names[0]}_features.npy')
labels = np.load(f'{names[0]}_labels.npy')
print(features.shape)

for s in names[1:]:
    feats = np.load(f'{s}_features.npy')
    lbs = np.load(f'{s}_labels.npy')
    print(feats.shape, features.shape)
    features = np.vstack((features, feats))
    labels = np.vstack((labels, lbs))