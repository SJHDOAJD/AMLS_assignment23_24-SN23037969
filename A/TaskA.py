from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import os

def load_path():

    # Use absolute path to define file path
    current_script_path = os.path.abspath(__file__)
    amls_dir_path = os.path.dirname(os.path.dirname(current_script_path))

    # create the dataset path
    datasets_path = os.path.join(amls_dir_path, 'Datasets', 'PneumoniaMNIST.npz')

    return datasets_path

def codeA():

    # Use the data path to load the original datasets
    datasets_path = load_path()
    data = np.load(datasets_path)

    # split the datasets for training, validation and test
    train_images_x = data['train_images']
    train_labels_y = data['train_labels']

    valid_images_x = data['val_images']
    valid_labels_y = data['val_labels']

    test_images_x = data['test_images']
    test_labels_y = data['test_labels']

    # define the size of image
    size = train_images_x[0].size

    def preprocessing(x,y):

        # reshape the image shape
        images = x.reshape(x.shape[0], size, )
        # change to 1D
        labels = y.ravel()

        return images,labels

    X_train, y_train = preprocessing(train_images_x, train_labels_y)
    X_val, y_val = preprocessing(valid_images_x, valid_labels_y)
    X_test, y_test = preprocessing(test_images_x, test_labels_y)

    # define the initial model of random forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    params = {
        'max_depth': [None,2,5,10,20],
        'min_samples_leaf': [1,5,10,20,50,100],
        'n_estimators': [10,50,100,150,170,200]
    }

    # Create the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=params, cv = 3, n_jobs=-1, verbose=1, scoring="accuracy")

    grid_search.fit(X_train, y_train)

    # Get the best score
    print("Best score of GridSearchCV: ", grid_search.best_score_)

    # Get the best parameters for random forest
    rf_best = grid_search.best_estimator_
    print("Best Estimator by GridSearchCV: ", grid_search.best_estimator_)

    # Create 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Get the validation and mean validation accuracy
    cv_scores = cross_val_score(rf_best, X_train, y_train, cv=kf, scoring='accuracy')
    mean_score = np.mean(cv_scores)

    print("Each 5-fold's accuracy: ", cv_scores)
    print("Average 5-Fold CV Accuracy:", mean_score)

    # Use the best parameters to train the random forest model to get the validation and test accuracy of Task A
    clf = rf_best
    clf.fit(X_train, y_train)

    # Get validation accuracy through validation dataset
    validation_pred = clf.predict(X_val)
    validation_accuracy = accuracy_score(y_val, validation_pred)
    print("Validation Accuracy for Task A: ", validation_accuracy)

    # Get test accuracy through test dataset
    test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print("Test Accuracy for Task A: ", test_accuracy)

if __name__ == "__main__":
    codeA()