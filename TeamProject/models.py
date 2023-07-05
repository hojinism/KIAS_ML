import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def make_baseline_model(input_shape, params):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(params[0], activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal', input_shape=input_shape))
    model.add(Conv2D(params[0], activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(params[1], activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
    model.add(Conv2D(params[1], activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(params[2], activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.2))
    model.add(Dense(params[3], activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='TruncatedNormal'))
    return model

def make_resnet50(input_shape):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model