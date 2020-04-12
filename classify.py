import numpy as np
import tensorflow as tf


N_CLASSES = 7 #10  # between 0 and 9

model = tf.keras.Sequential([
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(N_CLASSES)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])


def generate_data():
    while True:
        feats = np.load(f'MiniLille1_features.npy')
        lbs = np.load(f'MiniLille2_labels.npy')

        # shuffle data
        inds = np.random.shuffle(list(range(len(feats))))
        feats = feats[inds]
        lbs = lbs[inds]     
        
        bs = 64
        for i in range(len(feats) // bs):
            yield (feats[i*bs:(i+1)*bs], lbs[i*bs:(i+1)*bs])

model.fit(generate_data(),
                    steps_per_epoch=10000, epochs=10)

"""

names = ['Lille1_1', 'Lille1_2', 'Lille2', 'Paris1', 'Paris2']

for s in names:
    feats = np.load(f'{s}_features.npy')
    lbs = np.load(f'{s}_labels.npy')

    # shuffle data
    inds = np.random.shuffle(range(len(feats)))
    feats = feats[inds]
    lbs = lbs[inds]    
    
    # train on batches
    batch_size = 64
    
    print(np.min(lbs), np.max(lbs))
    print(feats.shape)
    
    print('starting train on batch')
    model.train_on_batch(feats, lbs)
    print('done')


    #features = np.vstack((features, feats))
    #labels = np.vstack((labels, lbs))

    del feats
    del lbs
"""


