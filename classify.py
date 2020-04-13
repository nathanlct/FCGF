import numpy as np
import tensorflow as tf


for voxel_size in [0.10]:#, 0.05, 0.10, 0.15, 0.20, 0.4, 0.7, 1.0]:
    print('----------------------------------------------')
    print('TRAINING WITH VOXEL SIZE ', voxel_size)
    print('----------------------------------------------')

    batch_size = 64

    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.01)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(7, activation='softmax', activity_regularizer=tf.keras.regularizers.l1(0.05))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])

    # only take reduced versions for training
    lille1_features = np.load(f'train_data/MiniLille1_features_reduced_{str(voxel_size)}.npy')
    lille2_features = np.load(f'train_data/MiniLille2_features_{str(voxel_size)}.npy')
    paris_features = np.load(f'train_data/MiniParis1_features_reduced_{str(voxel_size)}.npy')

    lille1_labels = np.load(f'train_data/MiniLille1_labels_reduced_{str(voxel_size)}.npy')
    lille2_labels = np.load(f'train_data/MiniLille2_labels_{str(voxel_size)}.npy')
    paris_labels = np.load(f'train_data/MiniParis1_labels_reduced_{str(voxel_size)}.npy')

    train_features = np.vstack((lille1_features, paris_features))
    train_labels = np.append(lille1_labels, paris_labels)

    test_features = lille2_features
    test_labels = lille2_labels

    n_train = len(train_features) - (len(train_features) % batch_size)
    train_features = train_features[:n_train]
    train_labels = train_labels[:n_train]

    n_test = len(test_features) - (len(test_features) % batch_size)
    test_features = test_features[:n_test]
    test_labels = test_labels[:n_test]

    print('train:', train_features.shape, train_labels.shape)
    print('test:', test_features.shape, test_labels.shape)

    for e in range(1):
        print('STARTING EPOCH', e)

        # shuffle dataset
        inds = list(range(len(train_features)))
        np.random.shuffle(inds)
        train_features = train_features[inds]
        train_labels = train_labels[inds]
        del inds

        print('Starting training')
        for i in range(len(train_features) // batch_size):
            if i % 10000 == 0:
                print(f'batch {i}/{len(train_features)//batch_size}')
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            model.train_on_batch(train_features[batch_start:batch_end], train_labels[batch_start:batch_end])
        print('Training ended')
        print('Epoch stats:')

        y_pred = np.array([])
        for i in range(len(test_features) // batch_size):
            if i % 10000 == 0:
                print(f'batch {i}/{len(test_features)//batch_size}')
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            y = model.predict_on_batch(test_features[batch_start:batch_end])
            y_pred = np.append(y_pred, np.argmax(y, axis=1))

        # stats       
        avg_precision = 0
        avg_recall = 0
        avg_FI = 0
        avg_IoU = 0

        for i in range(7):
            TP = np.count_nonzero(np.logical_and(y_pred == i, test_labels == i))
            TN = np.count_nonzero(np.logical_and(y_pred != i, test_labels != i))
            FP = np.count_nonzero(np.logical_and(y_pred == i, test_labels != i))
            FN = np.count_nonzero(np.logical_and(y_pred != i, test_labels == i))

            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN) if TP + FN != 0 else 0
            FI = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
            IoU = TP / (TP + FP + FN) if TP + FP + FN != 0 else 0

            print(f'{i}: {np.count_nonzero(y_pred == i)} predicted, {np.count_nonzero(test_labels == i)} total, TP={TP}, TN={TN}, FP={FP}, FN={FN}')
            print(f'\tprecision={precision}, recall={recall}, FI={FI}, IoU={IoU}')

            coef = np.count_nonzero(test_labels == i) / len(test_labels)
            avg_precision += coef * precision
            avg_recall += coef * recall
            avg_FI += coef * FI
            avg_IoU += coef * IoU

        print('Averaged stats:')
        print(f'\tPrecision={avg_precision}, recall={avg_recall}, FI={avg_FI}, IoU={avg_IoU}')
        print('\tAccuracy:', np.count_nonzero(y_pred == test_labels) / len(test_labels))

        print()
        print()


"""
Averaged stats:
        Precision=0.8759900347775749, recall=0.8798060002815531, FI=0.8719450399075971, IoU=0.7876702101301646
        Accuracy: 0.8798060002815532






"""