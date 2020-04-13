import numpy as np
import tensorflow as tf


for voxel_size in [0.10]:#, 0.05, 0.10, 0.15, 0.20, 0.4, 0.7, 1.0]:
    print('----------------------------------------------')
    print('TRAINING WITH VOXEL SIZE ', voxel_size)
    print('----------------------------------------------')

    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.01)),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(7, activation='softmax', activity_regularizer=tf.keras.regularizers.l1(0.05))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=1e-2),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),#from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    lille1_features = np.load(f'train_data/MiniLille1_features_reduced_{str(voxel_size)}.npy')
    lille2_features = np.load(f'train_data/MiniLille2_features_reduced_{str(voxel_size)}.npy')
    paris_features = np.load(f'train_data/MiniParis1_features_{str(voxel_size)}.npy')

    lille1_labels = np.load(f'train_data/MiniLille1_labels_reduced_{str(voxel_size)}.npy')
    lille2_labels = np.load(f'train_data/MiniLille2_labels_reduced_{str(voxel_size)}.npy')
    paris_labels = np.load(f'train_data/MiniParis1_labels_{str(voxel_size)}.npy')

    train_features = np.vstack((lille1_features, paris_features))
    train_labels = np.append((lille1_labels, paris_labels))

    test_features = lille2_features
    test_labels = lille2_labels

    print('train:', train_features.shape, train_labels.shape)
    print('test:', test_features.shape, test_labels.shape)

    # shuffle data
    inds = np.random.shuffle(list(range(len(train_features))))
    train_features = train_features[inds]
    train_labels = train_labels[inds]
    del inds

    print('train:', train_features.shape, train_labels.shape)
    print('test:', test_features.shape, test_labels.shape)

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

        print('IoU: ', iou)

        print()
        print()
        print()
