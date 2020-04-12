import numpy as np
import tensorflow as tf

use_small_data = True

if use_small_data:
    N_CLASSES = 7
    names = ['MiniLille1', 'MiniLille2', 'MiniParis1']
else:
    N_CLASSES = 10  # between 0 and 9
    names = ['Lille1_1', 'Lille1_2', 'Lille2', 'Paris1', 'Paris2']

# for voxel_size in [0.01, 0.05, 0.10, 0.15, 0.20, 0.4, 0.7, 1.0]:
#     print('----------------------------------------------')
#     print('TRAINING WITH VOXEL SIZE ', voxel_size)
#     print('----------------------------------------------')

model = tf.keras.Sequential([
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(N_CLASSES)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['sparse_categorical_accuracy'])


if use_small_data:
    # train_features = np.load(f'train_data/{names[0]}_features_reduced_{str(voxel_size)}.npy')
    # train_labels = np.load(f'train_data/{names[0]}_labels_reduced_{str(voxel_size)}.npy')
    train_features = np.load(f'{names[0]}_features.npy')
    train_labels = np.load(f'{names[0]}_labels.npy')

    for s in names[1:]:
        features = np.load(f'{s}_features_reduced.npy')
        labels = np.load(f'{s}_labels_reduced.npy')

        train_features = np.vstack((train_features, features))
        train_labels = np.append(train_labels, labels)


    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


    # shuffle data
    # inds = np.random.shuffle(list(range(len(features))))
    # features = features[inds][0]
    # labels = labels[inds][0]

    print('\n\n\n\nDATA LOADED:', train_features.shape, train_labels.shape)   
    
    model.fit(train_dataset, epochs=1)

