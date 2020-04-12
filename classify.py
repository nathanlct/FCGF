import numpy as np
import tensorflow as tf

use_small_data = True

if use_small_data:
    N_CLASSES = 7
    names = ['MiniLille1', 'MiniLille2', 'MiniParis1']
else:
    N_CLASSES = 10  # between 0 and 9
    names = ['Lille1_1', 'Lille1_2', 'Lille2', 'Paris1', 'Paris2']

model = tf.keras.Sequential([
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(N_CLASSES)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])


if use_small_data:
    features = np.load(f'{names[0]}_features.npy')
    labels = np.load(f'{names[0]}_labels.npy')

    for s in names[1:]:
        feats = np.load(f'{s}_features.npy')
        lbs = np.load(f'{s}_labels.npy')

        features = np.vstack((features, feats))
        labels = np.append(labels, lbs)

    # shuffle data
    inds = np.random.shuffle(list(range(len(features))))
    features = features[inds][0]
    labels = labels[inds] 

    print('\n\n\n\nDATA LOADED:', features.shape, labels.shape)   
     
    model.fit(features, labels)

