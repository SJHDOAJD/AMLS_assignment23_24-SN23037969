from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import random
import os

current_script_path = os.path.abspath(__file__)
amls_dir_path = os.path.dirname(os.path.dirname(current_script_path))
datasets_path = os.path.join(amls_dir_path, 'Datasets', 'PathMNIST.npz')
data = np.load(datasets_path)

train_images_xB = data['train_images']
train_labels_yB = data['train_labels']

valid_images_xB = data['val_images']
valid_labels_yB = data['val_labels']

test_images_xB = data['test_images']
test_labels_yB = data['test_labels']

image_width, image_height = 28, 28  
num_classes = 9

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(image_width, image_height, 3)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])

train_labels_yB_one_hot = to_categorical(train_labels_yB, num_classes)
valid_labels_yB_one_hot = to_categorical(valid_labels_yB, num_classes)
test_labels_yB_one_hot = to_categorical(test_labels_yB, num_classes)

history = model.fit(train_images_xB, train_labels_yB_one_hot, epochs=16, batch_size=64, validation_data=(valid_images_xB, valid_labels_yB_one_hot), shuffle=True)

test_loss, test_acc = model.evaluate(test_images_xB, test_labels_yB_one_hot, verbose=2)


predictions = model.predict(test_images_xB)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_yB_one_hot, axis=1)

conf_matrix = confusion_matrix(true_classes, predicted_classes)


accuracy = accuracy_score(true_classes, predicted_classes)

precision = precision_score(true_classes, predicted_classes, average=None)
precision_micro = precision_score(true_classes, predicted_classes, average='micro')
precision_macro = precision_score(true_classes, predicted_classes, average='macro')

recall = recall_score(true_classes, predicted_classes, average=None)
recall_micro = recall_score(true_classes, predicted_classes, average='micro')
recall_macro = recall_score(true_classes, predicted_classes, average='macro')

f1 = f1_score(true_classes, predicted_classes, average=None)
f1_micro = f1_score(true_classes, predicted_classes, average='micro')
f1_macro= f1_score(true_classes, predicted_classes, average='macro')

precision = np.append(precision, [precision_micro, precision_macro])
recall = np.append(recall, [recall_micro, recall_macro])
f1 = np.append(f1, [f1_micro, f1_macro])
classes = list(range(num_classes)) + ['Micro', 'Macro']

x = np.arange(len(classes))
width = 0.2