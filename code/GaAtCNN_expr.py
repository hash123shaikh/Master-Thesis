# -*- coding: utf-8 -*-
"""
Created on Tue May 29th 2023

@author: Hasan
"""

import keras
import keras.backend as K
from keras.layers import Input,Dropout, Flatten,Dense,MaxPooling1D,multiply, concatenate
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt

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
"""

num_of_filters = 25
epochs = 70
Sp_value = 0.95
acc_cvscores = []
Pr_cvscores = []
Sn_cvscores = []
Mcc_cvscores = []


"""
# This line initializes a variable `num_of_filters` with the value 25. 
  It is a hyperparameter that specifies the number of filters to use in a convolutional layer. 
  The number of filters determines the number of feature maps the layer will produce.

# This line initializes a variable `epochs` with the value 70. 
  In machine learning, an epoch refers to one complete pass through the entire training dataset during model training. 
  Therefore, this variable indicates that the model will be trained for 70 epochs.

# This line initializes a variable `Sp_value` with the value 0.95. 
  This value represent a specific threshold or criterion for a performance metric.   

# These lines initialize four empty lists: `acc_cvscores`, `Pr_cvscores`, `Sn_cvscores`, and `Mcc_cvscores`. 
  These lists are used to store performance scores or metrics during the model evaluation process. 
  The specific metrics they represent are accuracy, precision, sensitivity (recall), and Matthews correlation coefficient (MCC), respectively.

"""

path = 'F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC/gatedAtnExpOutput.csv'

def shows_result(path,arr):
	with open(path, 'w') as f: # Change the path to your local system
		for item_exp in arr:
			for elem in item_exp:
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

# load METABRIC EXPR dataset
dataset_exp = numpy.loadtxt("F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC/METABRIC_gene_exp_1980.txt", delimiter="\t")# Change the path to your local system

# split into input (X) and output (Y) variables
X_exp = dataset_exp[:,0:400]
Y_exp = dataset_exp[:,400]


"""
1. Fix Random Seed:

# This line sets the random seed to 1 for reproducibility of random processes using NumPy. 
  Setting the random seed ensures that the random numbers generated during the execution remain the same on different runs, making the results reproducible.

2. Load METABRIC CNV Dataset:  

# This line loads the METABRIC CNV dataset from the specified file path. 
  The dataset is assumed to be in a tab-separated format (`"\t"` is the delimiter). The dataset contains 1980 rows and 201 columns. 
  The first 400 columns represent the input features (X_exp), and the last column represents the output labels (Y_exp).

3. Split Input (X) and Output (Y) Variables:

# This code splits the loaded dataset into input (X_exp) and output (Y_exp) variables. 
  `X_exp` contains all the rows of the dataset and the first 400 columns, representing the features or independent variables. 
  `Y_exp` contains all the rows of the dataset and the last column, representing the labels or dependent variable.
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

for train_index, test_index in kfold.split(X_exp, Y_exp):
    print("*************************************************************",i,"th Fold **********************************************************************")
    i=i+1
	#Spliting the exp data set into training and testing
    x_train_exp, x_test_exp=X_exp[train_index],X_exp[test_index]	
    y_train_exp, y_test_exp = Y_exp[train_index],Y_exp[test_index] 	
    x_train_exp = numpy.expand_dims(x_train_exp, axis=2)
    x_test_exp = numpy.expand_dims(x_test_exp, axis=2)

    
    """
1. 10-fold Cross-Validation Loop:

# This loop implements 10-fold cross-validation using the previously initialized `StratifiedKFold` object (`kfold`). 
  It iterates over 10 splits of the data, where each split consists of training and testing indices.
    
2. Printing the Fold Number:

# This code prints the current fold number (`i`) to the console. 
  The `i` variable was initialized as 1 before the loop. 
  It's used to keep track of the current fold number during cross-validation.

3. Splitting the EXP Dataset:

# This code splits the exp dataset into training and testing sets for the current fold. 
  The training set (`x_train_exp`, `y_train_exp`) contains the data indexed by `train_index`, and the testing set (`x_test_exp`, `y_test_exp`) contains the data indexed by `test_index`.

4. Reshaping the EXP Data:

# The exp data (`x_train_exp` and `x_test_exp`) is reshaped to add an additional dimension of size 1. 
  This is done to match the expected shape of the input data for the 1D convolutional layers (`Conv1D`) in the CNN model.
    """

    # first exp CNN Model***********************************************************
    #init =initializers.glorot_normal(seed=1)
    bias_init =keras.initializers.Constant(value=1)
    main_input_exp = Input(shape=(400,1),name='Input')

    """
#   Let's explain the code for the first cnv CNN model step by step:    

5. `bias_init = keras.initializers.Constant(value=1)`: This line initializes a constant bias value of 1 using the `keras.initializers.Constant` initializer.
   The bias value will be used in the convolutional layers as part of the initialization process.

6. `main_input_exp = Input(shape=(400,1),name='Input')`: This line defines the input layer for the CNN model. 
   The `Input` function creates a Keras input tensor with the shape `(400,1)`. 
   The first value (400) represents the number of time steps (features), and the second value (1) represents the number of channels. 
   In this case, it seems like the model expects input data with 400 features and one channel (perhaps for time-series data).
    """

    conv_exp1 = Conv1D(filters=num_of_filters,kernel_size=2,strides=1,padding='same',name='Conv1D_exp1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp1)
    gatedAtnConv_exp1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp1)
    mult_1_1 = multiply([gatedAtnConv_exp1,conv_exp1])
    mult_1_1_1 = multiply([gatedAtnConv_exp1_1,conv_exp1])
    pooled_exp1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_exp1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    """
7. `conv_exp1 = Conv1D(filters=num_of_filters, kernel_size=1, strides=2, padding='same', name='Conv1D_exp1', kernel_initializer='glorot_uniform', bias_initializer=bias_init, activity_regularizer=l2(0.001))(main_input_exp)`: This line creates the first convolutional layer (`conv_cnv1`). 
   
   Here are the details:

# `Conv1D`: This function creates a 1D convolutional layer.

# `filters=num_of_filters`: This specifies the number of filters (or output channels) in the convolutional layer. 
                            `num_of_filters` is a variable that you defined earlier in the code and seems to be set to 25.

# `kernel_size=1`: This sets the size of the convolutional kernel to 1. 
                   Since the kernel size is 1, this convolutional layer performs a 1x1 convolution on the input data.

# `strides=2`: This sets the stride of the convolutional layer to 2. 
               The stride determines the step size at which the kernel slides over the input. 
               In this case, the kernel moves two steps at a time, leading to downsampling.
    
# `padding='same'`: This sets the padding mode to 'same', meaning the input is padded with zeros so that the output size matches the input size.

# `name='Conv1D_exp1'`: This assigns a name to the layer for identification.

# `kernel_initializer='glorot_uniform'`: This sets the weight initialization method for the convolutional layer. 
                                       `'glorot_uniform'` is an initializer that draws weights from a uniform distribution based on the Glorot uniform initializer.    
    
# `bias_initializer=bias_init`: This sets the bias initializer for the convolutional layer. 
                                The constant bias value of 0.1 is used for initialization, as defined earlier.

# `activity_regularizer=l2(0.001)`: This sets the L2 regularization with a coefficient of 0.001 for the layer's output.    
    

8. `gatedAtnConv_exp1 = Conv1D(filters=num_of_filters, kernel_size=1, strides=1, padding='same', name='GatedConv1D1', activation='relu', kernel_initializer='glorot_uniform', bias_initializer=bias_init, activity_regularizer=l2(0.001))(conv_exp1)`: This line creates a second convolutional layer (`gatedAtnConv_exp1`) with a similar setup as the previous layer, but with a few differences:

# `strides=1`: The stride is set to 1, which means no downsampling is performed, and the input size is preserved.

# `activation='relu'`: The layer uses the ReLU (Rectified Linear Unit) activation function, which introduces non-linearity to the model.

#  The convolutional kernel has a size of 1, so this layer performs a 1x1 convolution as well.    
    
    
9. `gatedAtnConv_exp1_1 = Conv1D(filters=num_of_filters, kernel_size=3, strides=1, padding='same', name='GatedConv1D1_1', activation='relu', kernel_initializer='glorot_uniform', bias_initializer=bias_init, activity_regularizer=l2(0.001))(conv_exp1)`: 
    This line creates a third convolutional layer (gatedAtnConv_exp1_1) with a similar setup to gatedAtnConv_exp1, but the kernel size is set to 3. 
    This layer also performs a 1x1 convolution, but it considers a slightly wider context of the input.    
    

10. `mult_1_1 = multiply([gatedAtnConv_exp1, conv_exp1])`: This line multiplies the output of `gatedAtnConv_exp1` and `conv_exp1` element-wise. 
     This is essentially introducing a gating mechanism where the output of `gatedAtnConv_exp1` modulates the output of the previous convolutional layer (`conv_exp1`).


11. `mult_1_1_1 = multiply([gatedAtnConv_exp1_1, conv_exp1])`: This line multiplies the output of `gatedAtnConv_exp1_1` and `conv_exp1` element-wise, introducing another gating mechanism with a slightly different context window.


12. `pooled_exp1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)`: This line applies max-pooling to the output of mult_1_1. 
     Max-pooling with a pool size of 2 and stride 1 is used to downsample the data and reduce the dimensionality.

     
13. `pooled_exp1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)`: This line applies max-pooling to the output of `mult_1_1_1`, creating another downsampled version of the data with a slightly different context window.
    
    
This part of the code defines the first part of a complex CNN model for processing exp data. 
The model consists of multiple convolutional layers with gating mechanisms and max-pooling, allowing the model to capture different context windows of the input data.
    """

    conv_exp2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='Conv1D_exp2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp2)
    gatedAtnConv_exp2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp2)
    mult_2_2 = multiply([gatedAtnConv_exp2,conv_exp2])
    mult_2_2_2 = multiply([gatedAtnConv_exp2_2,conv_exp2])
    pooled_exp2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_exp2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
    
    merged = concatenate([pooled_exp1,pooled_exp1_1,pooled_exp2,pooled_exp2_2],name='merge',axis=1)
    flat_exp = Flatten(name='Flatten')(merged)

    """
#   Let's break down the code for merging and flattening the outputs of the convolutional layers:

14. `merged = concatenate([pooled_exp1, pooled_exp2, pooled_exp1_1, pooled_exp2_2], name='merge', axis=1):
    This line creates a merged tensor by concatenating the outputs of four max-pooling layers: `pooled_exp1`, `pooled_exp2`, `pooled_exp1_1`, and `pooled_exp2_2`. 
    The concatenation is performed along the specified axis (`axis=1`), which means that the data from each layer is stacked horizontally.
    The resulting tensor `merged` now contains the combined features extracted from different context windows of the input data. This merging of features allows the model to capture information at various scales.

15. `flat_exp = Flatten(name='Flatten')(merged)`: This line applies a `Flatten` layer to the merged tensor. 
     The `Flatten` layer reshapes the input tensor into a one-dimensional vector while maintaining the order of elements. 
     This is necessary because the subsequent layers in the model require a one-dimensional input. 
     The name `'Flatten'` is assigned to this layer for identification purposes.

In summary, the code snippet you provided merges the outputs of multiple max-pooling layers, each capturing different context windows, and then flattens the merged tensor to prepare it for feeding into the fully connected layers of the model. 
This helps the model combine information from different scales and contexts while maintaining a suitable format for further processing.
    """

    dense_exp = Dense(150,name='dense_exp',activation='tanh',activity_regularizer=l2(0.01))(flat_exp)
    drop_final1 = Dropout(rate = 0.5)(dense_exp)
    dense_final2 = Dense(100,name='dense_final2',activation='tanh',activity_regularizer=l2(0.01))(drop_final1)
    dense_final3 = Dense(50,name='dense_final3',activation='tanh',activity_regularizer=l2(0.01))(dense_final2)
    output = Dense(1,activation='sigmoid')(dense_final3)
    model = Model(inputs=main_input_exp, outputs=output)
    plot_model(model, to_file='F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC/exp_gated_attention.png') # Change the path to your local system

    """
# This section of the code defines the fully connected layers of the EXP CNN model and assembles the complete model:

16. `dense_exp = Dense(150, name='dense_exp', activation='tanh', activity_regularizer=l2(0.001))(flat_exp)`:
    This line creates a fully connected (dense) layer with 150 units/neurons.
    It takes the flattened output (`flat_exp`) from the previous layer and applies the hyperbolic tangent (`tanh`) activation function.
    Additionally, L2 regularization with a coefficient of 0.001 is applied to the layer's output.

    
17. `drop_final1 = Dropout(rate=0.25)(dense_exp)`: This line adds a dropout layer to the model. 
    Dropout is a regularization technique that randomly sets a fraction of input units to zero during training, which helps prevent overfitting. 
    Here, a dropout rate of 0.25 is specified, meaning approximately 25% of the units will be dropped during training.


18. `dense_final2 = Dense(100, name='dense_final2', activation='tanh', activity_regularizer=l2(0.001))(drop_final1)`:
    This line adds another fully connected layer with 100 units/neurons, followed by the hyperbolic tangent activation function and L2 regularization.

         
19. `dense_final3 = Dense(50, name='dense_final3', activation='tanh', activity_regularizer=l2(0.001))(dense_final2)`:
    Similarly, this line adds another fully connected layer with 50 units/neurons, activation function, and L2 regularization.

      
20. `output = Dense(1, activation='sigmoid')(dense_final3)`:
    This line creates the final output layer of the model with a single neuron and a sigmoid activation function. 
    The sigmoid activation function is commonly used for binary classification tasks as it squashes the output between 0 and 1, representing the predicted probability of the positive class.

      
21. `model = Model(inputs=main_input_exp, outputs=output)`: Here, the `Model` class is used to create the final model. 
    The inputs argument is set to `main_input_exp`, which is the input layer defined earlier. 
    The outputs argument is set to `output`, which is the final output layer defined above. 
    This line effectively assembles the complete model architecture.

     
22. `plot_model(model, to_file='F:/Dissertations/TOPIC SELECTION/Research paper/Read & Useful/pr_7_SiGaAtCNN/code/SiGaAtCNNstackedRF-master/Data/METABRIC/clinical_gated_attention.png')`:
    This line uses the plot_model function to generate a visualization of the model architecture and save it as an image file. 
    The image will provide a graphical representation of how the different layers of the model are connected.

     
In summary, this section defines the fully connected layers of the EXP CNN model, including activation functions, dropout regularization, and output layer configuration. 
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

    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)
    model.fit(x_train1, y_train1, epochs=epochs, batch_size=8,validation_data=(x_val1,y_val1))	
	
    """
# A portion of the training data (x_train_exp, y_train_exp) is split further into a training set (x_train1, y_train1) and a validation set (x_val1, y_val1). 
  This validation set will be used to monitor the model's performance during training.

# The model is trained using the training data and evaluated on the validation set for the specified number of epochs.
    """

    scores = model.evaluate(x_test_exp, y_test_exp,verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    Sn_cvscores.append(scores[1] * 100)
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cvscores), numpy.std(Sn_cvscores)))

"""
24. Evaluating the Model on Test Data:

# The trained model is evaluated on the testing data (x_test_exp, y_test_exp). 
  The evaluation result is printed to the console, showing the sensitivity at the specified specificity level for the current fold. 
  The sensitivity score is then appended to the list Sn_cvscores.

# After the cross-validation loop completes, the code prints the average sensitivity and standard deviation of the sensitivity across all folds.
"""

X_exp=numpy.expand_dims(X_exp, axis=2)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense_final3").output)
intermediate_output = intermediate_layer_model.predict(X_exp)
shows_result(path,intermediate_output)

y_pred_prob = model.predict(X_exp)
fpr, tpr, thresholds = roc_curve(Y_exp, y_pred_prob,pos_label=1)


"""
This section of code performs the following tasks:

25. `X_exp = numpy.expand_dims(X_exp, axis=2)`: This line expands the dimensions of the `X_exp` dataset along the third axis. 
    It adds an additional dimension with size 1 to each data point. 
    This is done to match the input shape expected by the CNN model, which has a shape of `(number_of_samples, number_of_features, 1)`.

    
26. `intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense_final3").output)`:
    This line creates an intermediate model that takes the same input as the original `model` and produces the output of the layer named "dense_final3".
    This intermediate model is useful for extracting the activations of the "dense_final3" layer for each input sample.

    
27. `intermediate_output = intermediate_layer_model.predict(X_exp)`:
    This line uses the intermediate model to predict the output of the "dense_final3" layer for the entire `X_exp` dataset. 
    The `intermediate_output` will contain the activations (outputs) of the "dense_final3" layer for each input sample.

    
28. `shows_result(path, intermediate_output)`:
    This line calls the `shows_result` function to save the intermediate outputs (`intermediate_output`) to a file specified by the `path`. 
    This function was defined earlier in the code and seems to write the intermediate outputs to a text file.

    
29. `y_pred = model.predict(X_exp)`:
    This line uses the trained `model` to predict the outputs for the entire `X_exp` dataset. 
    `y_pred` will contain the predicted probabilities for each input sample.

    
30. `fpr, tpr, thresholds = roc_curve(Y_exp, y_pred, pos_label=1)`:
    This line computes the Receiver Operating Characteristic (ROC) curve by comparing the predicted probabilities (`y_pred`) to the true labels (`Y_exp`). 
    The `roc_curve` function returns false positive rates (`fpr`), true positive rates (`tpr`), and the corresponding thresholds for different classification thresholds. 
    The `pos_label` parameter specifies which class label should be treated as the positive class in the computation of the ROC curve.


In summary, this section of code processes the exp data through the model, extracts the activations of an intermediate layer ("dense_final3"), saves the intermediate outputs, computes the ROC curve for the model's predictions, and calculates the corresponding false positive rates, true positive rates, and thresholds for different classification thresholds.
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
plt.plot(fpr,tpr, 'r', label = 'Gated_Attention_CNN-Gene Expression = %0.3f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

"""
The code you provided is used to create and display a Receiver Operating Characteristic (ROC) curve for evaluating the performance of a classification model. Let's break down the code step by step:

# `roc_auc = auc(fpr, tpr)`: This line calculates the Area Under the Curve (AUC) for the ROC curve. 
  The `auc` function from the `sklearn.metrics` module is used to compute the AUC, which quantifies the overall performance of the model across different threshold values.

  
# `plt.plot(fpr, tpr, 'r', label='GaAtCNN-EXPR = %0.3f' % roc_auc)`: This line plots the ROC curve using the False Positive Rate (FPR) on the x-axis and the True Positive Rate (TPR) on the y-axis. 
The `'r'` argument specifies that the line should be red. 
The `label` parameter provides a label for the plot legend, including the calculated AUC value formatted with three decimal places.


# `plt.xlabel('1-Sp (False Positive Rate)')`: This line sets the label for the x-axis to "1-Sp (False Positive Rate)", indicating the false positive rate (FPR) complemented by 1.


# `plt.ylabel('Sn (True Positive Rate)')`: This line sets the label for the y-axis to "Sn (True Positive Rate)", indicating the true positive rate (TPR).


# `plt.title('Receiver Operating Characteristics')`: This line sets the title of the plot to "Receiver Operating Characteristics", describing the content of the plot.


#`plt.legend()`: This line adds a legend to the plot, displaying the label provided in the plot function.


# `plt.show()`: This line displays the plot.
"""

