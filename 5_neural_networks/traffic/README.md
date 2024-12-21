# Initial model
Starting with a simple architecture with a single convolutional layer, a flatten layer
and a dense layer for classification. This is simple and pretty quick to train so should
give a good baseline to try to improve on.
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```
Results are already pretty good for a first model:
```
333/333 - 1s - 3ms/step - accuracy: 0.9015 - loss: 1.1321
```

# Second model
Trying to prevent overfitting by making two changes to reduce loss without being too aggressive,
adding a MaxPooling layer and a relatively modest dropout layer to begin with. I'm keeping
the same number of filters because that seems to have produced good results from the outset.
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```
Results are okay - the loss is reduced, but accuracy has decreased slightly as well.
```
333/333 - 1s - 2ms/step - accuracy: 0.8991 - loss: 0.7529
```

# Third model
Fixing a warning about passing `input_shape` to the convolutional layer, and I added a second
convolutional layer with more filters to try to capture more complex features. Added a second
MaxPool layer to reduce dimensions, and keeping the dropout layer at the same level.
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```
Results are much better - accuracy is much higher, and the loss is greatly reduced as well
```
333/333 - 1s - 2ms/step - accuracy: 0.9727 - loss: 0.1232
```

# Fourth model
Adding a third convolutional model with even more filters to try to capture even higher order
features, and adding `padding=same` to preserve dimensions across the conv layers. Added an
additional dense layer as well, and a BatchNormalization layer after each conv layer to
hopefully decrease training time. Slightly increased the dropout rate to prevent
overfitting with the additional layers.
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```
Results suggests overfitting - training accuracy reached a max of 99% with a min loss of 0.03%,
but evaluated accuracy is only 95%.
```
333/333 - 3s - 9ms/step - accuracy: 0.9575 - loss: 0.1597
```

# Reducing overfitting
Removing a conv block and reducing the second dense layer size to reduce the complexity, and
adding more droupout layers with increasing rates to try to reduce overfitting. This model
should be faster to train and hopefully improve generalization.
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```
Results show a huge improvement, and training time was shorter with the removed layers.
```
333/333 - 2s - 5ms/step - accuracy: 0.9885 - loss: 0.0416
```

# Trying for max performance
Adding double convolution layers in each block and increased the filters to try to capture
more complex features and more feature variations. Also adding a larger additional dense
layer, and keeping the dropout filters as they are because they seem to be working well.
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```
Training this model took a lot longer, but performance is pretty great:
```
333/333 - 7s - 20ms/step - accuracy: 0.9968 - loss: 0.0111
```

# Trying for better efficiency
Removing some layers including the double convolutions, and reducing the conv layer filters
and kernel sizes to make the model faster to train and run. Performance was already good
without the additional dense layer, so going back to one and reducing the size. Dropout and
BatchNormalization seem to be working well as is.
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

    tf.keras.layers.Conv2D(32, (2, 2), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(96, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```
Much faster to train and run, while still keeping very high accuracy and low loss. This final
model is 1.69% less accurate than the 'max performance' one, but runs 7x faster which I think
is a worthwhile tradeoff in this case.
```
333/333 - 1s - 4ms/step - accuracy: 0.9799 - loss: 0.0671
```
