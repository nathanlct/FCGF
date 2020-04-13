import numpy as np
import tensorflow as tf

use_small_data = True

if use_small_data:
    N_CLASSES = 7
    names = ['MiniLille1', 'MiniLille2', 'MiniParis1']
else:
    N_CLASSES = 10  # between 0 and 9
    names = ['Lille1_1', 'Lille1_2', 'Lille2', 'Paris1', 'Paris2']

for voxel_size in [0.05]:#, 0.05, 0.10, 0.15, 0.20, 0.4, 0.7, 1.0]:
    print('----------------------------------------------')
    print('TRAINING WITH VOXEL SIZE ', voxel_size)
    print('----------------------------------------------')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(N_CLASSES)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=1e-2),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])


    if use_small_data:
        train_features = np.load(f'train_data/{names[0]}_features_reduced_{str(voxel_size)}.npy')
        train_labels = np.load(f'train_data/{names[0]}_labels_reduced_{str(voxel_size)}.npy')
        # train_features = np.load(f'{names[0]}_features.npy')
        # train_labels = np.load(f'{names[0]}_labels.npy')

        test_features = np.load(f'train_data/{names[0]}_features_0.1.npy')
        test_labels = np.load(f'train_data/{names[0]}_labels_0.1.npy')

        for s in names[1:]:
            features = np.load(f'train_data/{s}_features_reduced_{str(voxel_size)}.npy')
            labels = np.load(f'train_data/{s}_labels_reduced_{str(voxel_size)}.npy')

            t_features = np.load(f'train_data/{s}_features_0.1.npy')
            t_labels = np.load(f'train_data/{s}_labels_0.1.npy')
            # features = np.load(f'{s}_features.npy')
            # labels = np.load(f'{s}_labels.npy')

            train_features = np.vstack((train_features, features))
            train_labels = np.append(train_labels, labels)

            test_features = np.vstack((test_features, t_features))
            test_labels = np.append(test_labels, t_labels)

        # shuffle data
        inds = np.random.shuffle(list(range(len(train_features))))
        train_features = train_features[inds]
        train_labels = train_labels[inds]
        del inds
        del features
        del labels

        if train_features.shape[0] == 1:
            train_features = train_features[0]
        if train_labels.shape[0] == 1:
            train_labels = train_labels[0]

        for kk in range(10):
            print('EPOCH', kk)


            # BATCH_SIZE = 64
            # SHUFFLE_BUFFER_SIZE = 1000

            # train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
            # train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)



            print('\n\n\n\nDATA LOADED:', train_features.shape, train_labels.shape)   
            
            # model.fit(train_dataset, epochs=1, validation_data=train_dataset)

            batch_size = 64
            for i in range(len(train_features) // batch_size):
                if i % 10000 == 0:
                    print(f'batch {i}/{len(train_features)//batch_size}')
                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size
                model.train_on_batch(train_features[batch_start:batch_end], train_labels[batch_start:batch_end])
                if i == 0:
                    print('end i=0')


            print('start predict')
            y_out = model.predict(test_features, batch_size=64)
            y_pred = np.argmax(y_out, axis=1)
            print('ACCURACY: ', np.count_nonzero(y_pred == test_labels)/len(test_labels))

            # and the intersection over union as the intersection of the points assigned to class $i$ by the points belonging to class $i$, divided by their union
            # $$\text{IoU}_i = \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i + \text{FN}_i}$$

            
            print()
            iou = 0
            for i in range(7):
                print(f'{i}: {np.count_nonzero(y_pred == i)} predicted, {np.count_nonzero(np.logical_and(y_pred == i, test_labels == i))} correctly predicted, {np.count_nonzero(test_labels == i)} total')
                iou += (np.count_nonzero(test_labels == i) / len(test_labels)) * np.count_nonzero(np.logical_and(y_pred == i, test_labels == i)) / np.count_nonzero(np.logical_or(y_pred == i, test_labels == i))
            #iou /= 7

            print('IoU: ', iou)

            print()
            print()
            print()
"""
0: 143997 predicted, 8338 correctly predicted, 138056 total
1: 953760 predicted, 610057 correctly predicted, 823213 total
2: 72 predicted, 71 correctly predicted, 1007452 total
3: 0 predicted, 0 correctly predicted, 13948 total
4: 0 predicted, 0 correctly predicted, 12956 total
5: 0 predicted, 0 correctly predicted, 85304 total
6: 1970023 predicted, 976786 correctly predicted, 986923 total

0: 635481 predicted, 48619 correctly predicted, 138056 total
1: 1096063 predicted, 750530 correctly predicted, 823213 total
2: 606 predicted, 601 correctly predicted, 1007452 total
3: 0 predicted, 0 correctly predicted, 13948 total
4: 1 predicted, 0 correctly predicted, 12956 total
5: 0 predicted, 0 correctly predicted, 85304 total
6: 1335701 predicted, 957355 correctly predicted, 986923 total
"""