from A import TaskA
from B import TaskB
import matplotlib.pyplot as plt
import seaborn as sns

# Task A

print("Best score of GridSearchCV: ", TaskA.grid_search.best_score_)
print("Best Estimator by GridSearchCV: ", TaskA.grid_search.best_estimator_)

print("Each 5-fold's accuracy: ", TaskA.cv_scores)
print("Average 5-Fold CV Accuracy:", TaskA.mean_score)

print("Validation Accuracy for Task A: ", TaskA.validation_accuracy)
print("Test Accuracy for Task A: ", TaskA.test_accuracy)


#Task B

print("Test accuracy for Task B:", TaskB.test_acc)


fig, ax = plt.subplots(2,1)
ax[0].plot(TaskB.history.history['loss'], color='b', label="Training Loss")
ax[0].plot(TaskB.history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(TaskB.history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(TaskB.history.history['val_acc'], color='r',label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)


plt.figure(figsize=(6, 4))
sns.heatmap(TaskB.conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (Task B)')
plt.show()


fig1, ax1 = plt.subplots(figsize=(14, 6))
rects1 = ax1.bar(TaskB.x - TaskB.width, TaskB.precision, TaskB.width, label='Precision')
rects2 = ax1.bar(TaskB.x, TaskB.recall, TaskB.width, label='Recall')
rects3 = ax1.bar(TaskB.x + TaskB.width, TaskB.f1, TaskB.width, label='F1')

ax1.set_xlabel('Classes')
ax1.set_ylabel('Score')
ax1.set_title('Performance Metrics for 9 Classes (Task B)')
ax1.set_xticks(TaskB.x)
ax1.set_xticklabels(TaskB.classes)
ax1.set_ylim(0, 1.1)
ax1.legend(loc='upper right')
ax1.grid(True)
plt.show()