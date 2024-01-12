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

def load_path():

    # Use absolute path to define the file path
    current_script_path = os.path.abspath(__file__)
    amls_dir_path = os.path.dirname(os.path.dirname(current_script_path))
    # define the dataset path
    datasets_path = os.path.join(amls_dir_path, 'Datasets', 'PathMNIST.npz')

    return datasets_path

def codeB():

    # Use the data path to load the original datasets
    datasets_path = load_path()
    data = np.load(datasets_path)

    # split the datasets for training, validation and test
    train_images_xB = data['train_images']
    train_labels_yB = data['train_labels']

    valid_images_xB = data['val_images']
    valid_labels_yB = data['val_labels']

    test_images_xB = data['test_images']
    test_labels_yB = data['test_labels']

    # define the image size and number of classes
    image_width, image_height = 28, 28  
    num_classes = 9

    # Use the random seed to keep the result without drastically change
    seed_value = 42
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Set the structure of CNN
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

    # Set the parameter of compile
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])

    # Reshape the labels of dataset
    train_labels_yB_one_hot = to_categorical(train_labels_yB, num_classes)
    valid_labels_yB_one_hot = to_categorical(valid_labels_yB, num_classes)
    test_labels_yB_one_hot = to_categorical(test_labels_yB, num_classes)

    # Train the CNN model with training and validation datasets
    history = model.fit(train_images_xB, train_labels_yB_one_hot, epochs=16, batch_size=64, validation_data=(valid_images_xB, valid_labels_yB_one_hot), shuffle=True)

    # Get the test accuracy
    test_loss, test_acc = model.evaluate(test_images_xB, test_labels_yB_one_hot, verbose=2)
    print("Test accuracy for Task B:", test_acc)

    # plot the accuracy and loss values of training and validation with each epoch
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training Loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
    ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

    # make confusion matrix
    predictions = model.predict(test_images_xB)

    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels_yB_one_hot, axis=1)

    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix (Task B)')
    plt.show()

    # make performance metrics for each classes
    accuracy = accuracy_score(true_classes, predicted_classes)

    # define the values in normal, micro and macro state
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

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    rects1 = ax1.bar(x - width, precision, width, label='Precision')
    rects2 = ax1.bar(x, recall, width, label='Recall')
    rects3 = ax1.bar(x + width, f1, width, label='F1')

    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics for 9 Classes (Task B)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')
    ax1.grid(True)
    plt.show()

if __name__ == "__main__":
    codeB()