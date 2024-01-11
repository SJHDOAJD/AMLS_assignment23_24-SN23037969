from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import os

current_script_path = os.path.abspath(__file__)
amls_dir_path = os.path.dirname(os.path.dirname(current_script_path))
datasets_path = os.path.join(amls_dir_path, 'Datasets', 'PneumoniaMNIST.npz')
data = np.load(datasets_path)

train_images_x = data['train_images']
train_labels_y = data['train_labels']

valid_images_x = data['val_images']
valid_labels_y = data['val_labels']

test_images_x = data['test_images']
test_labels_y = data['test_labels']

size = train_images_x[0].size

X_train = train_images_x.reshape(train_images_x.shape[0], size, )
X_val = valid_images_x.reshape(valid_images_x.shape[0], size, )
X_test = test_images_x.reshape(test_images_x.shape[0], size, )

y_train = train_labels_y.ravel()
y_val = valid_labels_y.ravel()
y_test = test_labels_y.ravel()

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

params = {
    'max_depth': [None,2,5,10,20],
    'min_samples_leaf': [1,5,10,20,50,100],
    'n_estimators': [10,50,100,150,170,200]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 3,
                           n_jobs=-1, verbose=1, scoring="accuracy")

grid_search.fit(X_train, y_train)

grid_search.best_score_

rf_best = grid_search.best_estimator_

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(rf_best, X_train, y_train, cv=kf, scoring='accuracy')
mean_score = np.mean(cv_scores)

clf = rf_best
clf.fit(X_train, y_train)

validation_pred = clf.predict(X_val)
validation_accuracy = accuracy_score(y_val, validation_pred)

test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)