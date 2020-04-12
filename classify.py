import numpy as np


names = ['Lille1_1', 'Lille1_2', 'Lille2', 'Paris1', 'Paris2']

for s in names:
    feats = np.load(f'{s}_features.npy')
    lbs = np.load(f'{s}_labels.npy')
    print(feats.shape)
    #features = np.vstack((features, feats))
    #labels = np.vstack((labels, lbs))
    del feats
    del lbs