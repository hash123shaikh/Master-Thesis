# -*- coding: utf-8 -*-
"""
Created on Mon May  25th 2023

@author: Hasan
"""

import keras
import keras.backend as K
from keras.layers import Input, Dropout, Flatten, Dense, MaxPooling1D, Activation
from keras.layers import multiply, concatenate
from sklearn.model_selection import StratifiedKFold, train_test_split 
# demonstrate data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler 
import numpy

from sklearn.metrics import roc_curve, auc
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
import tensorflow as tf


"""
# `keras` is the Keras deep learning library that allows you to build and train neural networks easily.

# `keras.backend as K` provides access to Keras backend operations. 
   The backend can be TensorFlow, Theano, and this import allows you to perform operations specific to the backend.

# `Input, Dropout, Flatten, Dense, MaxPooling1D, Activation` are different types of Keras layers used to define the architecture of neural networks. 
   For example, `Dense` is a fully connected layer, and `Conv1D` is a one-dimensional convolutional layer.

# `multiply, concatenate` are functions for element-wise multiplication and concatenation of tensors, respectively. 
   They are used when you need to perform complex operations involving multiple layers or branches in your neural network.

# `StratifiedKFold, train_test_split` are functions used to split the data into training and testing sets. 
   StratifiedKFold is particularly useful for ensuring balanced class distributions in classification tasks.

# `MinMaxScaler` is a class from scikit-learn that helps in normalizing the data by scaling features to a specific range, usually [0, 1].

# `numpy` is the fundamental package for scientific computing with Python. 
   It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

# `roc_curve, auc` are functions for computing the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) for binary classification tasks. 
   These metrics are commonly used for evaluating the performance of binary classifiers.

# `Model` is a class in Keras that allows you to define and compile a neural network model by specifying its input and output layers.

# `plot_model` is a utility function from Keras that can be used to visualize the architecture of a Keras model as a graph.

# `l2` is a regularization function that applies L2 regularization to the neural network's weights. 
   Regularization helps prevent overfitting by penalizing large weight values.

# `Conv1D` is a one-dimensional convolutional layer used for processing sequential data, such as time series or text data.

# `matplotlib.pyplot as plt` is part of the matplotlib library and is used for plotting graphs and visualizations.

# `tensorflow as tf` is the TensorFlow deep learning library.
"""

num_of_filters=25
epochs = 25
Sp_value = 0.95
acc_cvscores = []
Pr_cvscores = []
Sn_cvscores = []
Mcc_cvscores = []


"""
# This line initializes a variable `num_of_filters` with the value 25. 
  It is a hyperparameter that specifies the number of filters to use in a convolutional layer. 
  The number of filters determines the number of feature maps the layer will produce.

# This line initializes a variable `epochs` with the value 25. 
  In machine learning, an epoch refers to one complete pass through the entire training dataset during model training. 
  Therefore, this variable indicates that the model will be trained for 25 epochs.

# This line initializes a variable `Sp_value` with the value 0.95. 
  This value represent a specific threshold or criterion for a performance metric.   

# These lines initialize four empty lists: `acc_cvscores`, `Pr_cvscores`, `Sn_cvscores`, and `Mcc_cvscores`. 
  These lists are used to store performance scores or metrics during the model evaluation process. 
  The specific metrics they represent are accuracy, precision, sensitivity (recall), and Matthews correlation coefficient (MCC), respectively.
"""

path = 'F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC/gatedAtnClnOutput.csv' # Change the path to your local system

def shows_result(path,arr):
	with open(path, 'w') as f: # Change the path to your local system
		for item_clinical in arr:
			for elem in item_clinical:
				f.write(str(elem)+',')
			f.write('\n')


"""
# This is a function definition called `shows_result`. The function takes two arguments: `path` and `arr`. 
  The purpose of this function is to save the results stored in the `arr` list to a file specified by the `path`.

# Inside the function, it opens a file for writing (`'w'` mode) with the provided `path` and then iterates over each item in the `arr` list. 
  For each item, it iterates over the elements (`elem`) in that item and writes each element followed by a comma (`,`) to the file. 
  After writing all elements of an item, it writes a newline character (`'\n'`) to start a new line in the file. 
  This process continues for each item in `arr`, effectively writing the entire `arr` list to the file.
"""

def nlrelu(t,label):
            if label=='nlrelu':
                return tf.log(tf.nn.relu(t)+1.)
            elif label=='selu':
                return tf.nn.selu(t)

"""
# This is another function definition called `nlrelu`. It takes two arguments: `t` and `label`. 
  The purpose of this function is to apply a non-linear activation function to the input tensor `t` based on the value of the `label`.

# If `label` is `'nlrelu'`, the function applies a non-linear variant of the ReLU (Rectified Linear Unit) activation function. 
  It takes the ReLU of `t` using `tf.nn.relu`, adds 1 to the result, and then takes the logarithm of the sum using `tf.log`. 
  This non-linear variant of ReLU is sometimes used in specific cases to prevent zero-valued outputs.

# If `label` is `'selu'`, the function applies the scaled exponential linear unit (SELU) activation function using `tf.nn.selu`. 
  SELU is another type of non-linear activation function.
"""

import keras.backend as K

def sensitivity_at_specificity(specificity, **kwargs):
    def Sn(labels, predictions):
        # Calculate true negatives (tn), false positives (fp), and false negatives (fn)
        tn = K.sum((1 - labels) * (1 - predictions))
        fp = K.sum((1 - labels) * predictions)
        fn = K.sum(labels * (1 - predictions))
        
        # Calculate specificity and sensitivity
        specificity_value = tn / (tn + fp + K.epsilon())
        sensitivity_value = tn / (tn + fn + K.epsilon())
        
        # Calculate the sensitivity at the given specificity level
        sensitivity_at_specificity = K.switch(K.less(K.abs(specificity - specificity_value), 0.001), sensitivity_value, 0.0)
        
        return sensitivity_at_specificity

    return Sn

"""
# The outer function `sensitivity_at_specificity` takes the `specificity` as an argument, which represents the desired specificity level at which we want to calculate sensitivity. 
  The `**kwargs` allows the function to accept additional keyword arguments if needed.

# Inside the outer function, there is an inner function `Sn`, which calculates the sensitivity at a specific specificity level given binary labels and model predictions.

# The inner function `Sn` first calculates true negatives (`tn`), false positives (`fp`), and false negatives (`fn`) using the binary labels and model predictions. 
  The true negatives are the cases where the model correctly predicts the negative class, false positives are cases where the model predicts positive but the true label is negative, and false negatives are cases where the model predicts negative but the true label is positive.

# Next, the function calculates the specificity and sensitivity. Specificity is the ratio of true negatives to the total number of actual negatives, and sensitivity is the ratio of true negatives to the total number of actual positives. 
  The `K.epsilon()` is a small constant added to avoid division by zero.

# The function then calculates the sensitivity at the desired specificity level (`specificity`) using the `K.switch` function. 
  If the absolute difference between the desired specificity level and the actual specificity (`specificity_value`) is less than 0.001, it returns the `sensitivity_value` as the sensitivity at the specific specificity level. 
  Otherwise, it returns 0.0.

# Finally, the inner function `Sn` returns the calculated sensitivity at the specific specificity level.

# The outer function `sensitivity_at_specificity` returns the inner function `Sn`. 
  This design allows you to create a sensitivity function for a specific specificity level by calling `sensitivity_at_specificity(specificity_level)`, and then you can use this function as a custom metric during model training or evaluation in Keras. 
  It provides a way to evaluate the model's sensitivity (recall) at a specified specificity level, which can be useful in certain applications where class imbalance is a concern, and you want to optimize sensitivity at a specific specificity threshold.
"""

# fix random seed for reproducibility
numpy.random.seed(1)

# load METABRIC Clinical dataset
dataset_clinical = numpy.loadtxt("F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC/METABRIC_clinical_1980.txt", delimiter="\t") # Change the path to your local system

# split into input (X) and output (Y) variables
X_clinical = dataset_clinical[:,0:25]
Y_clinical = dataset_clinical[:,25]


"""
1. Fix Random Seed:

# Random Seed: numpy.random.seed(1) is used to fix the random seed. 
  This ensures that any random operation, such as shuffling data before splitting into training and testing sets, can be replicated exactly in future runs. 
  It's a crucial step for reproducibility in scientific experiments and model evaluations.  

2. Load METABRIC Clinical Dataset:  

# This line loads the METABRIC Clinical dataset from the specified file path. 
  The dataset is assumed to be in a tab-separated format (`"\t"` is the delimiter). The dataset contains 1980 rows and 26 columns. 
  The first 25 columns represent the input features (X_clinical), and the last column represents the output labels (Y_clinical).

3. Split Input (X) and Output (Y) Variables:

# This code splits the loaded dataset into input (X_clinical) and output (Y_clinical) variables. 
  `X_clinical` contains all the rows of the dataset and the first 25 columns, representing the features or independent variables. 
  `Y_clinical` contains all the rows of the dataset and the last column, representing the labels or dependent variable.
"""

# 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cvscores = []
i=1


"""
# The code initializes a `StratifiedKFold` object called `kfold`. 
  `StratifiedKFold` is a cross-validation technique that ensures that the class distribution in each fold is approximately the same as the original dataset. 
  This is important when dealing with imbalanced datasets or when preserving the proportion of classes during cross-validation. 
  
  The parameters used are as follows:

  1.`n_splits=10`: The number of folds in the cross-validation. 
    In this case, it will create 10 folds, meaning the dataset will be divided into 10 parts.
  
  2.`shuffle=True`: This parameter determines whether to shuffle the data before splitting it into folds. 
    By setting it to `True`, the data will be randomly shuffled before creating the folds, ensuring that the order of the data does not impact the cross-validation results.
  
  3.`random_state=1`: The random seed for reproducibility. 
    Setting this parameter ensures that the random shuffling and fold generation process will be the same on different runs, making the results reproducible.

# An empty list `cvscores` is created. 
  This list will be used to store the evaluation scores obtained during each fold of cross-validation.    

# The variable `i` is set to 1. 
  This variable is likely used to keep track of the current fold number during the cross-validation process.
"""

# Initialize counters before the loop starts:
total_train_indices = 0
total_test_indices = 0

for train_index, test_index in kfold.split(X_clinical, Y_clinical):
    
    # print(train_index)
    # print(test_index)

    print(f"{i}th Fold - Training Indices: *****************************************")
    for index in train_index:
        print(index)

    print(f"{i}th Fold - Testing Indices: *****************************************")
    for index in test_index:
        print(index)
    
    
    """
   In this integrated code, each index of `train_index` and `test_index` for each fold will be printed on a new line.
   
   The print statements are also labeled to indicate whether the indices are for training or testing, and for which fold, making it easier to interpret the output.
   
   When you run this code, you'll see a list of indices for training and testing sets printed separately on new lines for each fold of your cross-validation process.
   """    
    total_train_indices += len(train_index)
    total_test_indices += len(test_index)

    """
    Increment these counters by the length of `train_index` and `test_index` in each iteration of your loop:
    """

    print(i,"th Fold *****************************************")
    i=i+1

    # Print the total count after the loop:
    print("Total number of training indices:", total_train_indices)
    print("Total number of testing indices:", total_test_indices)

	#Spliting the clinical data set into training and testing
    x_train_clinical, x_test_clinical=X_clinical[train_index],X_clinical[test_index]	
    y_train_clinical, y_test_clinical = Y_clinical[train_index],Y_clinical[test_index] 	
    
    print(x_train_clinical)
    print(x_test_clinical)
    print("Shape of x_train_clinical Before `expand_dims`:", x_train_clinical.shape)
    print("Shape of x_test_clinical Before `expand_dims`:", x_test_clinical.shape)
    
    x_train_clinical = numpy.expand_dims(x_train_clinical, axis=2)
    x_test_clinical = numpy.expand_dims(x_test_clinical, axis=2)


    """
    # The code lines you've provided are used to reshape your training and testing datasets (`x_train_clinical` and `x_test_clinical`) by adding an additional dimension. 
    
    # This is typically done to prepare data for input into models that expect inputs of a specific shape, like Convolutional Neural Networks (CNNs).

    Here's an explanation of how the output of this code will look like:

    # Before `expand_dims`
      
      # Suppose `x_train_clinical` and `x_test_clinical` are numpy arrays with a shape of `(n_samples, n_features)`. 
      
      # For example, if you have 100 samples and 25 features, the shape would be `(100, 25)`.

    # After `expand_dims`
      
      # After applying `numpy.expand_dims(..., axis=2)`, an additional dimension is added to each of these arrays.
      
      # The shape of `x_train_clinical` and `x_test_clinical` will become `(n_samples, n_features, 1)`. 
        Following the earlier example, the new shape would be `(100, 25, 1)`.
      
      # This change means that each feature vector is now treated as a 2D array (25x1) instead of a 1D array (25,).

    # Purpose
      
      # The reason for this reshaping is that certain types of neural networks, like 1D CNNs, expect data to have a specific number of dimensions (e.g., samples, time-steps, features).
      
      # By adding an additional dimension, each feature vector is treated as a sequence, which is a format suitable for convolutional operations in a CNN.
      
      # This transformation doesn't change the actual data but alters its shape to conform to the expected input format for certain types of neural network architectures.
    """

    print(x_train_clinical)
    print(x_test_clinical)
    print("Shape of x_train_clinical After `expand_dims`:", x_train_clinical.shape)
    print("Shape of x_test_clinical After `expand_dims`:", x_test_clinical.shape)
    
    """
    # To print the shape of `x_train_clinical` and `x_test_clinical` in Python using NumPy, you can use the `.shape` attribute of these numpy arrays. 
      Here's how you can do it:
    
    # This code will output the dimensions of each array. The `.shape` attribute of a numpy array returns a tuple representing the dimensions of the array. 
      For instance, if `x_train_clinical` has 100 samples, 25 features, and you've expanded its dimensions as previously mentioned, it will show something like `(100, 25, 1)`.

    # Place these print statements in your code after the lines where `x_train_clinical` and `x_test_clinical` are defined and reshaped. 
      This will give you a clear understanding of how their shapes look before and after the transformation.     
    """
# *****************************************************************************
    """
1. Stratified K-Fold Cross-Validation

# `for train_index, test_index in kfold.split(X_clinical, Y_clinical):`

  # This is a `for` loop that iterates over the folds of a stratified k-fold cross-validation.
  
  # `kfold.split(X_clinical, Y_clinical)` generates indices for splitting the dataset into training and testing sets. For each fold, it provides a different set of indices.
  
  # `train_index` and `test_index` are arrays of indices for training and testing sets, respectively, for each fold.

  # Stratified k-fold ensures that each fold of the cross-validation process has approximately the same percentage of samples of each target class as the complete set.
    
2. Printing the Fold Number:

# This code prints the current fold number (`i`) to the console. 
  The `i` variable was initialized as 1 before the loop. 
  It's used to keep track of the current fold number during cross-validation.

3. Splitting the Clinical Dataset:

# `x_train_clinical, x_test_clinical = X_clinical[train_index], X_clinical[test_index]`
  
  # This line splits the input features (`X_clinical`) into training and testing sets using the indices (`train_index`, `test_index`).
  
  # `x_train_clinical` will contain the input features for the training set of the current fold.
  
  # `x_test_clinical` will contain the input features for the testing set of the current fold.

# `y_train_clinical, y_test_clinical = Y_clinical[train_index], Y_clinical[test_index]`
  
  # Similarly, this line splits the output labels (`Y_clinical`) into training and testing sets.
  
  # `y_train_clinical` will contain the labels for the training set of the current fold.
  
  # `y_test_clinical` will contain the labels for the testing set of the current fold.  
  
4. Reshaping the Clinical Data for CNN Input:

# `x_train_clinical = numpy.expand_dims(x_train_clinical, axis=2)`
  
  # This line adds an additional dimension to the `x_train_clinical` array. It's reshaping the data to make it suitable for input into a CNN.
  
  # The `expand_dims` function is used to add an extra dimension at the specified axis. Here, `axis=2` means the new dimension is added as the third dimension.
  
  # After this operation, each instance in `x_train_clinical` is not just a flat array of features but a 2D array where one dimension is of size 1. This is a standard practice when preparing data for 1D CNNs.

# `x_test_clinical = numpy.expand_dims(x_test_clinical, axis=2)`
  
  # This does the same for the testing set.      
    """
    
    # first Clinical Gated Attention CNN Model***********************************************************

    bias_init = keras.initializers.Constant(value=0.1)
    main_input_clinical = Input(shape=(25,1),name='Input')# for METABRIC data
    

# Let's explain the code for the first Clinical CNN model step by step:
  
    """
  The code snippet is part of setting up a CNN using the Keras library. 
  It involves initializing the bias parameter and defining the input layer of the network. 
  Let's break down these lines for a detailed understanding:

5. `Bias Initialization`

  # `bias_init = keras.initializers.Constant(value=0.1)`:

    # `What It Does`: This line of code initializes a constant bias value that will be used in the neural network layers.

    # `Initializer Function: keras.initializers.Constant` is a function from Keras that initializes a tensor to a constant value.

    # `Bias Value`: Here, the constant value is set to 0.1. 
       The bias is a parameter in many neural network layers (like Dense or Convolutional layers) that allows the model to fit the data better.

    # `Purpose`: Initializing biases to a small, non-zero value can help prevent nodes from being inactive at the start. 
       If bias is initialized to zero, it may slow down the learning process, especially in networks with activation functions like ReLU.

6. `Input Layer Definition`

  # `main_input_clinical = Input(shape=(25,1),name='Input')`:
    
    # `What It Does`: This line defines the input layer of the CNN model.
    
    # `Input Shape`: The `shape=(25,1)` argument specifies the shape of the input data that the model will receive.
    
    # The first dimension `25` represents the number of features for each sample. 
      In the context of clinical data, these could be different clinical measurements, or any other numerical features.
    
    # The second dimension `1` indicates the number of channels in the data. 
      For standard tabular data or time series, this is usually set to 1. 
      In image data, this would correspond to the color channels (e.g., 3 for RGB).
    
    # `Input Function`: The `Input` function from Keras is used to instantiate a Keras tensor, which serves as the starting layer of the neural network. 
       It's where the data enters the network for processing.

    # `Naming the Layer`: The `name='Input'` argument assigns a name to this input layer. 
       Naming layers can be helpful in more complex models where you need to specify connections or debug the model.

7. `Overall Context`
  
  # These steps are part of the initial setup of a CNN. 
    You're preparing the network to receive input data of a certain shape and setting initial conditions (like bias values) that can impact the learning process.
    """

    conv_clinical1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=2,padding='same',name='Conv1D_clinical1',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(main_input_clinical)
    print("Shape of conv_clinical1 layer :",conv_clinical1.shape)
    gatedAtnConv_clinical1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical1)
    print("Shape of conv_clinical1 layer :",gatedAtnConv_clinical1.shape)
    gatedAtnConv_clinical1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical1)
    print("Shape of conv_clinical1 layer :",gatedAtnConv_clinical1_1.shape)
    mult_1_1 = multiply([gatedAtnConv_clinical1,conv_clinical1])
    mult_1_1_1 = multiply([gatedAtnConv_clinical1_1,conv_clinical1])
    pooled_clinical1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_clinical1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)


    """
7. `conv_clinical1 = Conv1D(filters=num_of_filters, kernel_size=1, strides=2, padding='same', name='Conv1D_clinical1', kernel_initializer='glorot_uniform', bias_initializer=bias_init, activity_regularizer=l2(0.001))(main_input_clinical)`
    This line defines a 1D convolutional layer named `(conv_clinical1)`. It takes `main_input_clinical` as its input.
    Here are the details:

# `Conv1D`: This function creates a 1D convolutional layer.

# `filters=num_of_filters`:  This specifies the number of output filters in the convolutional layer. 
                             Each filter convolves with the input to produce a different feature map. 
                             If `num_of_filters` is set to 25, this layer will produce 25 feature maps.

# `kernel_size=1`: The kernel size of 1 means that the convolution operation is performed with a window of size 1. 
                   This essentially means that the convolution operation is applied to each individual element, without considering its neighbors.
                   
# `strides=2`: The stride determines the step size at which the kernel slides over the input.
               A stride of 2 indicates that the convolution window will move two steps at a time across the input.
               This results in downsampling and reduces the size of the output feature maps by half compared to the input size.

# `padding='same'`: This sets the padding mode to 'same', meaning the input is padded with zeros so that the output size matches the input size.

# `name='Conv1D_clinical1'`: This assigns a name to the layer for identification.

# `kernel_initializer='glorot_uniform'`: This sets the weight initialization method for the convolutional layer. 
                                       `'glorot_uniform'` is an initializer that draws weights from a uniform distribution based on the Glorot uniform initializer.
    
# `bias_initializer=bias_init`: This sets the bias initializer for the convolutional layer. 
                                The constant bias value of 0.1 is used for initialization, as defined earlier.

# `activity_regularizer=l2(0.001)`: This sets the L2 regularization with a coefficient of 0.001 for the layer's output.    
    

8. `gatedAtnConv_clinical1 = Conv1D(filters=num_of_filters, kernel_size=1, strides=1, padding='same', name='GatedConv1D1', activation='relu', kernel_initializer='glorot_uniform', bias_initializer=bias_init, activity_regularizer=l2(0.001))(conv_clinical1)`
    This line creates a second convolutional layer (`gatedAtnConv_clinical1`) with a similar setup as the previous layer, but with a few differences:

# `strides=1`: The stride is set to 1, which means no downsampling is performed, and the input size is preserved.

# `activation='relu'`: The layer uses the ReLU (Rectified Linear Unit) activation function, which introduces non-linearity to the model.

#  The convolutional kernel has a size of 1, so this layer performs a 1x1 convolution as well.    
    
    
9. `gatedAtnConv_clinical1_1 = Conv1D(filters=num_of_filters, kernel_size=3, strides=1, padding='same', name='GatedConv1D1_1', activation='relu', kernel_initializer='glorot_uniform', bias_initializer=bias_init, activity_regularizer=l2(0.001))(conv_clinical1)` 
    This line creates a third convolutional layer (gatedAtnConv_clinical1_1) with a similar setup to gatedAtnConv_clinical1, but the kernel size is set to 3.
    This layer also performs a 1x1 convolution, but it considers a slightly wider context of the input.    
    

10. `mult_1_1 = multiply([gatedAtnConv_clinical1, conv_clinical1])`: 
     This line multiplies the output of `gatedAtnConv_clinical1` and `conv_clinical1` element-wise. 
     This is essentially introducing a gating mechanism where the output of `gatedAtnConv_clinical1` modulates the output of the previous convolutional layer `(conv_clinical1)`.


11. `mult_1_1_1 = multiply([gatedAtnConv_clinical1_1, conv_clinical1])`: 
     This line multiplies the output of `gatedAtnConv_clinical1_1` and `conv_clinical1` element-wise, introducing another gating mechanism with a slightly different context window.


12. `pooled_clinical1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)`: 
     This line applies max-pooling to the output of mult_1_1. 
     Max-pooling with a pool size of 2 and stride 1 is used to downsample the data and reduce the dimensionality.

     
13. `pooled_clinical1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)`: 
     This line applies max-pooling to the output of `mult_1_1_1`, creating another downsampled version of the data with a slightly different context window.
    
    
This part of the code defines the first part of a complex CNN model for processing clinical data. 
The model consists of multiple convolutional layers with gating mechanisms and max-pooling, allowing the model to capture different context windows of the input data.
    """

    conv_clinical2 = Conv1D(filters=num_of_filters,kernel_size=2,strides=2,padding='same',name='Conv1D_clinical2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(main_input_clinical)
    gatedAtnConv_clinical2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical2)
    gatedAtnConv_clinical2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical2)
    mult_2_2 = multiply([gatedAtnConv_clinical2,conv_clinical2])
    mult_2_2_2 = multiply([gatedAtnConv_clinical2_2,conv_clinical2])
    pooled_clinical2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_clinical2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
    
    
    merged = concatenate([pooled_clinical1, pooled_clinical2,pooled_clinical1_1, pooled_clinical2_2],name='merge',axis=1)
    flat_clinical = Flatten(name='Flatten')(merged)
    
    """
#   Let's break down the code for merging and flattening the outputs of the convolutional layers:

14. `merged = concatenate([pooled_clinical1, pooled_clinical2, pooled_clinical1_1, pooled_clinical2_2], name='merge', axis=1):
    This line creates a merged tensor by concatenating the outputs of four max-pooling layers: `pooled_clinical1`, `pooled_clinical2`, `pooled_clinical1_1`, and `pooled_clinical2_2`. 
    The concatenation is performed along the specified axis (`axis=1`), which means that the data from each layer is stacked horizontally.
    The resulting tensor `merged` now contains the combined features extracted from different context windows of the input data. This merging of features allows the model to capture information at various scales.

15. `flat_clinical = Flatten(name='Flatten')(merged)`: This line applies a `Flatten` layer to the merged tensor. 
     The `Flatten` layer reshapes the input tensor into a one-dimensional vector while maintaining the order of elements. 
     This is necessary because the subsequent layers in the model require a one-dimensional input. 
     The name `'Flatten'` is assigned to this layer for identification purposes.

In summary, the code snippet you provided merges the outputs of multiple max-pooling layers, each capturing different context windows, and then flattens the merged tensor to prepare it for feeding into the fully connected layers of the model. 
This helps the model combine information from different scales and contexts while maintaining a suitable format for further processing.
    """
    
    dense_clinical = Dense(150,name='dense_clinical',activation='tanh',activity_regularizer=l2(0.001))(flat_clinical)
    drop_final1 = Dropout(rate = 0.25)(dense_clinical)
    dense_final2 = Dense(100,name='dense_final2',activation='tanh',activity_regularizer=l2(0.001))(drop_final1)
    dense_final3 = Dense(50,name='dense_final3',activation='tanh',activity_regularizer=l2(0.001))(dense_final2)
    output = Dense(1,activation='sigmoid')(dense_final3)    
    model = Model(inputs=main_input_clinical, outputs=output)
    plot_model(model, to_file='F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC/clinical_gated_attention.png') # Change the path to your local system
    
    """
# This section of the code defines the fully connected layers of the Clinical Gated Attention CNN model and assembles the complete model:

16. `dense_clinical = Dense(150, name='dense_clinical', activation='tanh', activity_regularizer=l2(0.001))(flat_clinical)`:
    
    This line creates a fully connected (dense) layer with 150 units/neurons.
    It takes the flattened output (`flat_clinical`) from the previous layer and applies the hyperbolic tangent (`tanh`) activation function.
    Additionally, L2 regularization with a coefficient of 0.001 is applied to the layer's output.

    
17. `drop_final1 = Dropout(rate=0.25)(dense_clinical)`: 
    
    This line adds a dropout layer to the model. 
    Dropout is a regularization technique that randomly sets a fraction of input units to zero during training, which helps prevent overfitting. 
    Here, a dropout rate of 0.25 is specified, meaning approximately 25% of the units will be dropped during training.

     
18. `dense_final2 = Dense(100, name='dense_final2', activation='tanh', activity_regularizer=l2(0.001))(drop_final1)`:
    
    This line adds another fully connected layer with 100 units/neurons, followed by the hyperbolic tangent activation function and L2 regularization.

     
19. `dense_final3 = Dense(50, name='dense_final3', activation='tanh', activity_regularizer=l2(0.001))(dense_final2)`:
    
    Similarly, this line adds another fully connected layer with 50 units/neurons, activation function, and L2 regularization.

      
20. `output = Dense(1, activation='sigmoid')(dense_final3)`:
    
    This line creates the final output layer of the model with a single neuron and a sigmoid activation function. 
    The sigmoid activation function is commonly used for binary classification tasks as it squashes the output between 0 and 1, representing the predicted probability of the positive class.

      
21. `model = Model(inputs=main_input_clinical, outputs=output)`: 
    
    Here, the Model class is used to create the final model. 
    The inputs argument is set to main_input_clinical, which is the input layer defined earlier. 
    The outputs argument is set to output, which is the final output layer defined above. 
    This line effectively assembles the complete model architecture.

     
22. `plot_model(model, to_file='F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC/clinical_gated_attention.png')`:
    This line uses the plot_model function to generate a visualization of the model architecture and save it as an image file. 
    The image will provide a graphical representation of how the different layers of the model are connected.

     
In summary, this section defines the fully connected layers of the Clinical Gated Attention CNN model, including activation functions, dropout regularization, and output layer configuration.
It then assembles the complete model using the Model class and visualizes the model architecture for reference.    
    """

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])
    
    """
23. Compiling and Training the Model:

`model`: This refers to the Keras model that was defined earlier. 
         It represents the neural network model that you built using the functional API.

`compile()`: This method is used to configure the model for training. 
             It takes three important arguments:

`loss`: This specifies the loss function that will be used to measure how well the model is performing. 
        In this case, the loss function is set to `'binary_crossentropy'`. 
        Binary cross-entropy is commonly used for binary classification problems, where the goal is to minimize the difference between predicted and actual class labels for each sample.

`optimizer`: This specifies the optimization algorithm to use during the training process. 
             Here, the optimizer is set to `'Adam'`. 
             Adam (short for Adaptive Moment Estimation) is a popular optimization algorithm that adapts the learning rate during training to achieve faster convergence and better generalization.

`metrics`: This specifies the list of evaluation metrics that will be used to monitor the model's performance during training and evaluation. 
           In this case, the model uses a custom metric called `sensitivity_at_specificity(Sp_value)`. 
           The `Sp_value` is a sensitivity value set earlier in the code, representing the desired specificity level for evaluating the sensitivity (recall) of the model.

# The `sensitivity_at_specificity` is a custom metric that you defined earlier in your code. 
  It is a function that calculates the sensitivity at a specific specificity level given binary labels and model predictions. 
  By passing it to the `metrics` argument, you instruct Keras to calculate this metric and display its value during training and evaluation.

To summarize, `model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])` sets up the model for training using binary cross-entropy as the loss function, Adam as the optimization algorithm, and sensitivity at a specific specificity level as the evaluation metric.         
    """
    
    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
    model.fit(x_train1, y_train1, epochs=epochs, batch_size=8,validation_data=(x_val1,y_val1))	
    
    """
# A portion of the training data (x_train_clinical, y_train_clinical) is split further into a training set (x_train1, y_train1) and a validation set (x_val1, y_val1). 
  This validation set will be used to monitor the model's performance during training.

# The model is trained using the training data and evaluated on the validation set for the specified number of epochs.
    """

    scores = model.evaluate(x_test_clinical, y_test_clinical,verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    Sn_cvscores.append(scores[1] * 100)
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cvscores), numpy.std(Sn_cvscores)))

"""
24. Evaluating the Model on Test Data:

# The trained model is evaluated on the testing data (x_test_clinical, y_test_clinical). 
  The evaluation result is printed to the console, showing the sensitivity at the specified specificity level for the current fold. 
  The sensitivity score is then appended to the list Sn_cvscores.

# After the cross-validation loop completes, the code prints the average sensitivity and standard deviation of the sensitivity across all folds.
"""

X_clinical = numpy.expand_dims(X_clinical, axis=2)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense_final3").output)
intermediate_output = intermediate_layer_model.predict(X_clinical)
shows_result(path,intermediate_output)

y_pred = model.predict(X_clinical)
fpr, tpr, thresholds = roc_curve(Y_clinical, y_pred,pos_label=1)

"""
This section of code performs the following tasks:

25. `X_clinical = numpy.expand_dims(X_clinical, axis=2)`: 
    
    This line expands the dimensions of the `X_clinical` dataset along the third axis. 
    It adds an additional dimension with size 1 to each data point. 
    This is done to match the input shape expected by the CNN model, which has a shape of `(number_of_samples, number_of_features, 1)`.

    
26. `intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense_final3").output)`:
    
    This line creates an intermediate model that takes the same input as the original `model` and produces the output of the layer named "dense_final3".
    This intermediate model is useful for extracting the activations of the "dense_final3" layer for each input sample.

    
27. `intermediate_output = intermediate_layer_model.predict(X_clinical)`:
    
    This line uses the intermediate model to predict the output of the "dense_final3" layer for the entire `X_clinical` dataset. 
    The `intermediate_output` will contain the activations (outputs) of the "dense_final3" layer for each input sample.

    
28. `shows_result(path, intermediate_output)`:
    
    This line calls the `shows_result` function to save the intermediate outputs (`intermediate_output`) to a file specified by the `path`. 
    This function was defined earlier in the code and seems to write the intermediate outputs to a csv file.

    
29. `y_pred = model.predict(X_clinical)`:
    
    This line uses the trained `model` to predict the outputs for the entire `X_clinical` dataset. 
    `y_pred` will contain the predicted probabilities for each input sample.

    
30. `fpr, tpr, thresholds = roc_curve(Y_clinical, y_pred, pos_label=1)`:
    
    This line computes the Receiver Operating Characteristic (ROC) curve by comparing the predicted probabilities (`y_pred`) to the true labels (`Y_clinical`). 
    The `roc_curve` function returns false positive rates (`fpr`), true positive rates (`tpr`), and the corresponding thresholds for different classification thresholds. 
    The `pos_label` parameter specifies which class label should be treated as the positive class in the computation of the ROC curve.

In summary, this section of code processes the clinical data through the model, extracts the activations of an intermediate layer ("dense_final3"), saves the intermediate outputs, computes the ROC curve for the model's predictions, and calculates the corresponding false positive rates, true positive rates, and thresholds for different classification thresholds.
"""

def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)

"""
This section of code defines a function `evaluate_threshold` and then uses it to evaluate sensitivity and specificity at different classification thresholds.

31. `def evaluate_threshold(threshold)`: This line defines a function named `evaluate_threshold` that takes a single argument `threshold`. 
    Inside the function, the following actions are performed:

# `print('Sensitivity:', tpr[thresholds > threshold][-1])`: This line calculates and prints the sensitivity (true positive rate) at the given threshold. 
  It uses the `tpr` array, which contains true positive rates computed from the ROC curve, and the `thresholds` array, which contains the corresponding thresholds. 
  The sensitivity is calculated for the last threshold that is greater than the specified `threshold`.

# `print('Specificity:', 1 - fpr[thresholds > threshold][-1])`: This line calculates and prints the specificity (true negative rate) at the given threshold. 
  It uses the `fpr` array, which contains false positive rates computed from the ROC curve, and the `thresholds` array. 
  The specificity is calculated as 1 minus the false positive rate for the last threshold that is greater than the specified `threshold`.


32. `for threshold in numpy.arange(0, 1, 0.05)`: This line sets up a loop that iterates over a range of thresholds from 0 to 1 with a step size of 0.05.

# `print('********* Threshold = ', threshold, ' ************')`: Within the loop, this line prints a header indicating the current threshold being evaluated.

# `evaluate_threshold(threshold)`: This line calls the evaluate_threshold function with the current threshold as an argument. This evaluates and prints the sensitivity and specificity at the given threshold.

In summary, this section of code defines a function to evaluate sensitivity and specificity at a specified threshold and then loops through a range of thresholds, printing the results for each threshold. 
This process helps analyze the model's performance at different decision thresholds and can provide insights into the trade-off between sensitivity and specificity.
"""

roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr, 'r', label = 'Gated_Attention_CNN-Clinical = %0.3f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

"""
The code you provided is used to create and display a Receiver Operating Characteristic (ROC) curve for evaluating the performance of a classification model. Let's break down the code step by step:

# `roc_auc = auc(fpr, tpr)`: This line calculates the Area Under the Curve (AUC) for the ROC curve. 
  The `auc` function from the `sklearn.metrics` module is used to compute the AUC, which quantifies the overall performance of the model across different threshold values.

  
# `plt.plot(fpr, tpr, 'r', label='SiGaAtCNN-CLN = %0.3f' % roc_auc)`: This line plots the ROC curve using the False Positive Rate (FPR) on the x-axis and the True Positive Rate (TPR) on the y-axis. 
The `'r'` argument specifies that the line should be red. 
The `label` parameter provides a label for the plot legend, including the calculated AUC value formatted with three decimal places.


# `plt.xlabel('1-Sp (False Positive Rate)')`: This line sets the label for the x-axis to "1-Sp (False Positive Rate)", indicating the false positive rate (FPR) complemented by 1.


# `plt.ylabel('Sn (True Positive Rate)')`: This line sets the label for the y-axis to "Sn (True Positive Rate)", indicating the true positive rate (TPR).


# `plt.title('Receiver Operating Characteristics')`: This line sets the title of the plot to "Receiver Operating Characteristics", describing the content of the plot.


#`plt.legend()`: This line adds a legend to the plot, displaying the label provided in the plot function.


# `plt.show()`: This line displays the plot.
"""
