# Machine-Learning-Short-Projects

## Neural Network

**Dataset:**
- **_Name:_** Iris Dataset
- **_Source:_** https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
- **_Number of Class Labels:_** 3 [Iris Setosa] [Iris Versicolour] [Iris Virginica]
- **_Number of Attributes:_** 4
- **_Class Distribution: 33.3% for each of 3 classes._**
- **_All attributes are continuous._**
- **_None Missing Attribute Values._**

**Code:**

For calculating the test accuracy we have predicted the class labels and checked with the target class label to see if that matches. The total matches are divide by the total test data to find the accuracy of the training model. Also, the results shown above are on the scale of 0 to 1 where 1 signifies the highest value. 

Methods like LabelEncoder and OneHotEncoder are used to convert categorical data into float values so that the neural network can be trained. Moreover, as per the requirement, section of code has been included to handle missing values.


**Output:**

|Activation Function|Test Size|Average Error on Training Data|Average Error on Test Data|Accuracy on Test Data|
|---|---|---|---|---|
|Sigmoid Function|25%|0.0372|0.0399|0.9210|
|Sigmoid Function|30%|0.0376|0.0399|0.9333|
|tanh Function|25%|0.3332|0.3508|0.6315|
|tanh Function|30%|0.1743|0.2691|0.8444|
|ReLu Function|25%|0.1666|0.1666|0.3421|
|ReLu Function|30%|0.1666|0.1666|0.3556|

**Results:**

For this Iris dataset as we can observe sigmoid function performs better than tanh and ReLu. Depending upon network whether it is deeper or shallow, the accuracy of activation function might differ. For example, ReLu is only good for deeper network and hence the accuracy of the test data is less compared to other two in our network. Tanh activation performs better for shallow network compared ReLu and hence its accuracy is better than ReLu but not better than sigmoid activation functions.


## Hyper Parameter Tuning

**Dataset:**
- **_Name:_** Wine recognition data
- **_Source:_** https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
- **_Number of Class Labels:_** 3 [Class 1 - 59] [Class 2 - 71] [Class 3 - 48]
- **_Number of Attributes:_** 13
- **_All attributes are continuous._**
- **_None Missing Attribute Values._**

**Code:**

In order to change the running algorithm user need to enter the value as per the menu list provided on the console at runtime. Here is the list:
1. DecisionTree
2. NeuralNet
3. SVM                  (Support Vector Machine)
4. GNB                  (Gaussian Naive Bayes)
5. LR                   (Logistic Regression)
6. knearest             (k - Nearest Neighbor)
7. Bagging 
8. RandomForest
9. AdaBoost
10. GBC                 (Gradient Boosting Classifier)
11. XGBoost 

The user input should match the list names and these are case sensitive.

**Output:**

| Algorithm |Tuned Parameters|Avg. Precision|Avg. Recall|Avg. F1|Acuuracy Score|
|---|---|---|---|---|---|
| Decision Tree |{'min_weight_fraction_leaf': 0.1, 'max_depth': 5, 'max_features': 7, 'min_samples_leaf': 10, 'max_leaf_nodes': 70}|0.85|0.84|0.83|0.8333
|Neural Net|{'hidden_layer_sizes': (100, 20), 'activation': 'logistic', 'max_iter': 500, 'learning_rate': 'constant','solver': 'adam'}|0.97|0.97|0.97|0.9722|
|Support Vector Machine|{'C': 1, 'gamma': 0.001, 'kernel':'linear', ‘degree’: 3}|0.96|0.97|0.97|0.9722|
|Gaussian Naive Bayes|{'priors': [.3,.4,.3] }|1.0|1.0|1.0|1.0|
|Logistic Regression|{'multi_class': 'multinomial', 'C': 1, 'max_iter': 100, 'solver': 'newton-cg', 'tol': 0.0001, 'penalty': 'l2'}|0.94|0.94|0.94|0.9444|
|k - Nearest Neighbor|{'p': 1, 'weights': 'distance', 'algorithm': 'ball_tree', 'n_neighbors': 10}|0.83|0.83|0.83|0.8333|
|Bagging|{'n_estimators': 100, 'random_state': 10, 'max_features': 10, 'max_samples': 4}|0.93|0.92|0.92|0.9167|
|Random Forest|{'max_depth': 14, 'n_estimators': 20, 'max_features': 5, 'criterion': 'gini'}|0.97|0.97|0.97|0.9722|
|AdaBoost Classifier|{'algorithm': 'SAMME.R', 'random_state': 4, 'learning_rate': 1.5, 'n_estimators': 100}|0.97|0.96|0.97|0.9722|
|Gradient Boosting Classifier|{'max_depth': 1, 'learning_rate': 1.0, 'n_estimators': 20, 'loss': 'deviance'}|1.0|1.0|1.0|1.0|
|XGBoost Classifier|{'booster': 'gblinear', 'learning_rate': 1.0, 'max_delta_step': 1, 'n_estimators': 100, 'seed': None}|0.94|0.94|0.94|0.9444|


**Results:**

Gaussian Bayes and Gradient Boosting Classifier provided the best accuracy on Test data with the parameters as per the above table. The reason for Gaussian Bayes to perform well as compared to other algorithm is because the attributes were independent from each other and this helps reduce correlation and redundancy in data, eventually leading to a much better accuracy. Gradient Boosting Classifier performed well because the algorithm uses weak classifiers and gradually increases accuracy by introducing new weak learners which compensate the weaknesses of existing weak learners.

To increase the efficiency of the model further pre-processing like normalization or mean can be applied on the dataset before training the model. Moreover, it is also effective to reduce all the dependencies on attributes to have a clean dataset which would help increase the accuracy.
