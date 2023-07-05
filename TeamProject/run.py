import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
# GPU settings
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="baseline", type=str, help="model alias")
parser.add_argument("--channel", default=2, type=int, help="number of channels")
parser.add_argument("--init_lr", default=1e-3, type=float, help="initial learning rates")
parser.add_argument("--epochs", default=20, type=int, help="number of epochs")
parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
args = parser.parse_args()

# data preprocess
print(f"@@@@ Load dataset...")
WORKDIR = os.getenv("WORKDIR")
filename = f"{WORKDIR}/Particle_Images/data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5"
data1 = h5py.File(filename, "r")
Y1 = data1["y"][:200000]
X1 = data1["X"][:200000]
filename = f"{WORKDIR}/Particle_Images/data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5"
data0 = h5py.File(filename, "r")
Y0 = data0["y"][:200000]
X0 = data0["X"][:200000]
X_final = np.concatenate((X0[:], X1[:]), axis=0)
Y_final = np.concatenate((Y0[:], Y1[:]), axis=0)
print(f"@@@@ X_final: {X_final.shape}")
print(f"@@@@ Y_final: {Y_final.shape}")

input_shape = (32, 32, args.channel)    # 1 for using Hit-Energy channel only
x_train, x_test, y_train, y_test = train_test_split(
    X_final[:, :, :, 0:args.channel],
    Y_final,
    test_size=0.2,
    random_state=42
)
print(f"@@@@ x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"@@@@ x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# baseline model
model = tf.keras.models.Sequential()
model.add(Conv2D(16, activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal', input_shape=input_shape))
model.add(Conv2D(16, activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
model.add(Conv2D(32, activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='TruncatedNormal'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_initializer='TruncatedNormal'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', kernel_initializer='TruncatedNormal'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.init_lr), metrics=['accuracy'])
model.summary()

# start training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-4)
checkpoint_path = f"./checkpoints/checkpoints_{args.name}"
checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True
)
history=model.fit(x_train, y_train,\
        batch_size=args.batch_size,\
        epochs=args.epochs,\
        validation_split=0.1,\
        callbacks=[reduce_lr, checkpoint_callback],\
        shuffle=True)

# Evaluate on test set
score = model.evaluate(x_test, y_test, verbose=1)
print('\nTest loss / accuracy: %0.4f / %0.4f'%(score[0], score[1]))
y_pred = model.predict(x_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('Test ROC AUC:', roc_auc)

plt.plot([0, 1], [0, 1], 'k--')
#plt.legend(loc=2, prop={'size': 15})
plt.plot(fpr, tpr, label='Model 1 (ROC-AUC = {:.3f})'.format(roc_auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(f"plots/ROC_{args.name}.png")
