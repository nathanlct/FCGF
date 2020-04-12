import numpy as np
import tensorflow as tf


N_CLASSES = 10  # between 0 and 9

model = tf.keras.Sequential([
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(N_CLASSES)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

names = ['Lille1_1', 'Lille1_2', 'Lille2', 'Paris1', 'Paris2']

for s in names:
    feats = np.load(f'{s}_features.npy')
    lbs = np.load(f'{s}_labels.npy')
    print(np.min(lbs), np.max(lbs))
    print(feats.shape)
    
    print('starting train on batch')
    model.train_on_batch(feats, lbs)
    print('done')


    #features = np.vstack((features, feats))
    #labels = np.vstack((labels, lbs))
    
    del feats
    del lbs