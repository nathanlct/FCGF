import numpy as np
import tensorflow as tf

use_small_data = True

if use_small_data:
    N_CLASSES = 7
    names = ['MiniLille1', 'MiniLille2', 'MiniParis1']
else:
    N_CLASSES = 10  # between 0 and 9
    names = ['Lille1_1', 'Lille1_2', 'Lille2', 'Paris1', 'Paris2']

for voxel_size in [0.10]:#, 0.05, 0.10, 0.15, 0.20, 0.4, 0.7, 1.0]:
    print('----------------------------------------------')
    print('TRAINING WITH VOXEL SIZE ', voxel_size)
    print('----------------------------------------------')

    model = tf.keras.Sequential([
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='tanh'),
        #tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(N_CLASSES)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])


    if use_small_data:
        train_features = np.load(f'train_data/{names[0]}_features_reduced_{str(voxel_size)}.npy')
        train_labels = np.load(f'train_data/{names[0]}_labels_reduced_{str(voxel_size)}.npy')
        # train_features = np.load(f'{names[0]}_features.npy')
        # train_labels = np.load(f'{names[0]}_labels.npy')

        for s in names[1:]:
            features = np.load(f'train_data/{s}_features_reduced_{str(voxel_size)}.npy')
            labels = np.load(f'train_data/{s}_labels_reduced_{str(voxel_size)}.npy')
            # features = np.load(f'{s}_features.npy')
            # labels = np.load(f'{s}_labels.npy')

            train_features = np.vstack((train_features, features))
            train_labels = np.append(train_labels, labels)

        # shuffle data
        inds = np.random.shuffle(list(range(len(train_features))))
        train_features = train_features[inds]
        train_labels = train_labels[inds]

        if train_features.shape[0] == 1:
            train_features = train_features[0]
        if train_labels.shape[0] == 1:
            train_labels = train_labels[0]


        BATCH_SIZE = 64
        SHUFFLE_BUFFER_SIZE = 1000

        train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)



        print('\n\n\n\nDATA LOADED:', train_features.shape, train_labels.shape)   
        
        model.fit(train_dataset, epochs=1, validation_data=train_dataset)

        y_out = model.predict(train_features, batch_size=64)
        y_pred = np.argmax(y_out, axis=1)
        for i in range(7):
            print(f'{i}: {np.count_nonzero(y_pred == i)} predicted, {np.count_nonzero(np.logical_and(y_pred == i, train_labels == i))} correctly predicted, {np.count_nonzero(train_labels == i)} total')