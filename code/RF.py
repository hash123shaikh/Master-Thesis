# -*- coding: utf-8 -*-
"""
Created on Sat May 30th 2023

@author: Hasan
"""

from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import fbeta_score, make_scorer
import keras.backend as K
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import keras
import warnings
warnings.filterwarnings("ignore")

"""
# `from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier`: These are classes from the scikit-learn library that implement ensemble learning methods. 
  `AdaBoostClassifier` is an implementation of the AdaBoost algorithm, and `RandomForestClassifier` is an implementation of the random forest algorithm.

# `from sklearn.tree import DecisionTreeClassifier`: This class is part of scikit-learn and represents a decision tree classifier.

# `import numpy as np`: This imports the NumPy library and assigns it the alias "np". 
   NumPy is used for numerical computations and array operations.

# `from sklearn.model_selection import cross_validate`: The `cross_validate` function from scikit-learn is used for performing cross-validation on a dataset.

# `from sklearn.metrics import fbeta_score, make_scorer`: `fbeta_score` is a metric from scikit-learn that calculates the F-beta score, which is a combination of precision and recall. 
  `make_scorer` is used to create a scoring function that can be used in model evaluation.

# `import keras.backend as K`: This imports the backend functions of Keras, a deep learning framework. 
   It is used for accessing low-level operations and functions.

# `from sklearn.metrics import confusion_matrix`: The `confusion_matrix` function from scikit-learn is used to compute a confusion matrix to evaluate classification performance.

# `from sklearn.model_selection import train_test_split, StratifiedKFold`: These classes are used for splitting datasets into train and test sets, as well as performing stratified k-fold cross-validation.

# `from sklearn.base import BaseEstimator, ClassifierMixin`: These are base classes that can be used to create custom estimators and classifiers in scikit-learn.

# `import os`: The `os` module provides a way to use operating system-dependent functionality, such as interacting with the file system.

# `import keras`: This imports the Keras library, a high-level deep learning framework that runs on top of TensorFlow or Theano. 
   It is used for building and training neural network models.
"""

from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import keras
import keras.backend as K
from keras.layers import Input,Dropout, Flatten,Dense, Activation,MaxPooling1D
from keras.layers import dot, multiply, concatenate
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils.vis_utils import plot_model
# from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

"""
# `from sklearn.metrics import roc_curve, auc`: These functions are used to calculate the Receiver Operating Characteristic (ROC) curve and the corresponding Area Under the Curve (AUC), which are common metrics for evaluating classification models.

# `import matplotlib.pyplot as plt`: The `matplotlib.pyplot` module provides functions for creating various types of plots and visualizations.

# `from sklearn.model_selection import cross_val_score, cross_val_predict`: These functions are used for performing cross-validation and obtaining cross-validated scores and predictions.

# `from sklearn import metrics`: The `metrics` module from scikit-learn contains various metrics for evaluating model performance, such as accuracy, precision, recall, etc.

# `from keras.layers import Input, Dropout, Flatten, Dense, Activation, MaxPooling1D`: These are different layers provided by Keras for building neural network models. 
  They are used to define the architecture of the model.

# `from keras.layers import dot, multiply, concatenate`: These layers are used for various operations in neural network architectures, such as dot product, element-wise multiplication, and concatenation.

# `from keras.utils.vis_utils import plot_model`: This function is used to visualize the architecture of a Keras model.

# `from keras.regularizers import l2`: The l2 regularizer from Keras is used to apply L2 regularization to the model's weights.

# `from keras.layers.convolutional import Conv1D`: This is a Keras layer for one-dimensional convolution, which is commonly used in Convolutional Neural Networks (CNNs).

# `from sklearn.metrics import classification_report, confusion_matrix`: The `classification_report` function generates a text report of various classification metrics, and `confusion_matrix` computes a confusion matrix to evaluate the performance of a classification model.
"""

path = "F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC"
file1 = "gatedAtnAll_Input.csv"
dataset1 = np.loadtxt(os.path.join(path, file1),delimiter=",")
X1= dataset1[:,0:775]
Y1 = dataset1[:,775]

"""
1. `path = "F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC"`
   This line assigns a file `path` to the variable path, which indicates the directory where your data files are located.


2. `file1 = "gatedAtnAll_Input.csv" & file2 = "STACKED_RF.csv"` 
   These lines define the names of two CSV files that contain your input features and labels for your datasets.


3. `dataset1 = np.loadtxt(os.path.join(path, file1), delimiter=",")` 
   This line reads the CSV file named `gatedAtnAll_Input.csv` from the specified directory using NumPy's `loadtxt` function. 
   It loads the data into a NumPy array `dataset1` and uses , as the delimiter to separate values in the CSV.

   
4. `X1 = dataset1[:, 0:755] & Y1 = dataset1[:, 775]` 
   These lines extract the input features and labels from `dataset1`. 
   `X1` contains the input features, and it includes all rows and the first 450 columns of `dataset1`.
   `Y1` contains the corresponding labels, and it includes all rows and the 451st column of `dataset1`.
"""

rfc = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0,class_weight='balanced')
scores1 = cross_val_score(rfc, X1, Y1, cv=10,verbose=0)
print ("Cross-validated scores:", scores1)
print("Accuracy = %.3f%% (+/- %.3f%%)\n" % (np.mean(scores1), np.std(scores1)))

"""
This code snippet involves the creation and evaluation of an ensemble model using the Random Forest algorithms. Let's break it down step by step:

5. `rfc = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0, class_weight='balanced')`:

   Here, a `RandomForestClassifier` is created. This classifier is an ensemble model that consists of multiple decision trees (forest). The specified parameters include:

# `n_estimators=200`: Number of trees in the forest.

# `max_depth=None`: Maximum depth of each tree. If set to `None`, nodes are expanded until they contain less than `min_samples_split` samples.

# `random_state=0`: Random seed for reproducibility.

# `class_weight='balanced'`: Adjusts the class weights to balance the distribution of classes in the training data.


6. `scores1 = cross_val_score(rf, X1, Y1, cv=10, verbose=0)`: 
    The `cross_val_score` function is used to perform cross-validation on the ensemble model (`rf`) using the input features (`X1`) and labels (`Y1`). 

# `cv=10` specifies a 10-fold cross-validation. 

# The `scores1` variable stores the array of accuracy scores obtained for each fold.


7. `print("Cross-validated scores:", scores1)`: 
    This line prints the array of cross-validated accuracy scores obtained from each fold of the cross-validation.


8. `print("Accuracy = %.3f%% (+/- %.3f%%)\n" % (np.mean(scores1), np.std(scores1)))`: 
    This line calculates and prints the mean and standard deviation of the cross-validated accuracy scores. 
    It provides an estimate of the model's accuracy along with a measure of its variability across different folds.


9. How Training and Testing are Handled
   
#  Cross-Validation (cross_val_score):

# In each fold of cross-validation, a subset of X1 and Y1 is used as the training set, and another subset as the validation set. 
  This process is repeated 10 times (as cv=10), each time with a different split of training and validation data.

# The model (rfc) is trained on the training subset and evaluated on the validation subset for each fold. 
  The accuracy of the model on the validation set is recorded.
  
# This approach ensures that the model's performance is evaluated on different subsets of the data, giving a more robust assessment of its generalizability.    
   
10. Summary

# In summary, this code segment trains and evaluates an ensemble model using the Random Forest algorithm as the base estimator. 
  Cross-validation is used to assess the accuracy of the model, and the mean accuracy along with its variability is printed as the final output.
"""

predictions1 = cross_val_predict(rfc, X1, Y1, cv=10,method='predict_proba')[:, 1]
fpr, tpr, thresholds = roc_curve(Y1, predictions1,pos_label=1)
roc_auc = auc(fpr, tpr)

def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    print('AUC:', roc_auc)

evaluate_threshold(0.45)

for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)

plt.plot(fpr,tpr, 'c', label = 'Multimodal Gated Attention CNN- {CLN, EXPR, CNA} = %0.3f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

"""
# This code segment involves predicting class probabilities, generating an ROC curve, and evaluating a specific threshold. 
  Let's break it down step by step:

10. `predictions1 = cross_val_predict(rfc, X1, Y1, cv=10, method='predict_proba')[:, 1]`:
     
     This line uses the `cross_val_predict` function to generate class probabilities for the positive class (class 1) using the trained ensemble model `rfc`. 
     
     The parameter `method='predict_proba'` indicates that probabilities are being predicted. 
     
     The `[:, 1]` indexing extracts the probabilities for the positive class.

    
11. `fpr, tpr, thresholds = roc_curve(Y1, predictions1, pos_label=1)`:
     
     The `roc_curve` function calculates the False Positive Rate (FPR), True Positive Rate (TPR), and associated thresholds for different probability thresholds. 
     
     It uses the true labels `Y1` and the predicted probabilities `predictions1`. 
    
    `pos_label=1` specifies that class 1 is considered the positive class.

    
12. `roc_auc = auc(fpr, tpr)`: 
     This line calculates the Area Under the Curve (AUC) for the ROC curve using the FPR and TPR values.


13. The following lines define a function `evaluate_threshold(threshold)` that evaluates and prints sensitivity, specificity, and AUC for a given threshold:


# `print('Sensitivity:', tpr[thresholds > threshold][-1])`: 
   
   Prints the sensitivity (True Positive Rate) at the specified threshold. 
  
  `tpr[thresholds > threshold]` extracts TPR values corresponding to thresholds greater than the specified threshold. 
  
  `[-1]` indexes the last value in this array, which corresponds to the sensitivity at that threshold.

  
# `print('Specificity:', 1 - fpr[thresholds > threshold][-1])`: 
   
   Prints the specificity (1 - False Positive Rate) at the specified threshold. 
   
   Similarly to sensitivity, this line calculates specificity based on the FPR values.

   
# `print('AUC:', roc_auc)`: 
   
   Prints the calculated AUC for the ROC curve.


14. `evaluate_threshold(0.45)`: This line calls the `evaluate_threshold` function with a threshold value of 0.45, which means it evaluates and prints sensitivity, specificity, and AUC for the ROC curve at that specific threshold.


In summary, this code segment predicts class probabilities using cross-validation, generates an ROC curve, and evaluates sensitivity, specificity, and AUC at a specified threshold. 
It provides insights into the performance of the ensemble model at a specific decision threshold.
"""
