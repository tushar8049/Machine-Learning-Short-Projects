# This example has been taken from SciKit documentation and has been
# modifified to suit this assignment. You are free to make changes, but you
# need to perform the task asked in the lab assignment


from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import pandas as pd

print(__doc__)

# Loading the Digits dataset
wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")

"""
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
"""
print("1. DecisionTree \n2. NeuralNet \n3. SVM                  (Support Vector Machine) \n4. GNB                  (Gaussian Naive Bayes)")
print("5. LR                   (Logistic Regression) \n6. knearest             (k - Nearest Neighbor) \n7. Bagging")
print("8. RandomForest \n9. AdaBoost \n10. GBC                 (Gradient Boosting Classifier) \n11. XGBoost")
algo = input("Enter the algorithm you wanna run: ")

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(wine)
ncols = len(wine.columns)
nrows = len(wine.index)
X = wine.iloc[:, 1:(ncols)].values.reshape(nrows, ncols - 1)
y = wine.iloc[:, 0].values.reshape(nrows, 1)

# Split the dataset in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# This is a key step where you define the parameters and their possible values
# that you would like to check.
"""
tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}
                    ]
"""
# We are going to limit ourselves to accuracy score, other options can be
# seen here:
# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Some other values used are the predcision_macro, recall_macro
scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    """
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)
    """
    if algo == "SVM":
        tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000], 'degree': [3,1,5,10,7],
                             'random_state': [1, 2, 5, 10, 12]}]
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s' % score)
        clf.fit(X_train, y_train)

    elif algo == "DecisionTree":
        tuned_parameters = [{'min_samples_leaf': [10,20,30,40], 'max_depth': [4,5,8,9,10,13],
                             'max_features': [4,6,7,8,9,10,13], 'max_leaf_nodes': [45,55,65,70],
                             'min_weight_fraction_leaf': [0.1, 0.2, 0.3, 0.4, 0.5]}]
        dt=DecisionTreeClassifier()
        clf = GridSearchCV(dt, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "NeuralNet":
        tuned_parameters = [{'hidden_layer_sizes': [(100,20),(50,50),(80,80)], 'activation': ['relu', 'logistic', 'identity'],
                             'solver': ['lbfgs', 'adam'], 'learning_rate': ['constant', 'adaptive'], 'max_iter': [400, 500, 200]}]
        mlp=MLPClassifier()
        clf = GridSearchCV(mlp, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "GNB":
        tuned_parameters = [{'priors': [[0.35,0.35,0.3],[.25,.25,.5],[.3,.4,.3],[.5,.25,.25]]}]
        gnb=GaussianNB()
        clf = GridSearchCV(gnb, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "LR":
        tuned_parameters = [{'penalty':['l2'], 'C': [1, 10, 100, 1000], 'tol': [1e-4],
                             'solver': ['newton-cg', 'lbfgs', 'sag'],
                             'max_iter': [100,200,400], 'multi_class': ['ovr', 'multinomial']}]
        lr=LogisticRegression()
        clf = GridSearchCV(lr, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "knearest":
        tuned_parameters = [{'algorithm':['ball_tree','kd_tree', 'brute'], 'n_neighbors': [5, 10, 25, 100],
                             'weights': ['uniform', 'distance'], 'p': [1,2,3]}]
        knc=KNeighborsClassifier()
        clf = GridSearchCV(knc, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "Bagging":
        tuned_parameters = [{'max_features':[1,2,3,4,10], 'n_estimators': [5, 10, 20, 25, 50, 100],
                             'random_state': [1,2,5,10,12], 'max_samples': [1,2,3,4]}]
        bc=BaggingClassifier()
        clf = GridSearchCV(bc, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "RandomForest":
        tuned_parameters = [{'max_depth':[1,2,4,10,14], 'n_estimators': [10, 20, 50, 100, 200, 400],
                             'criterion': ['gini', 'entropy'], 'max_features': [5,8,1,11]}]
        rfc=RandomForestClassifier()
        clf = GridSearchCV(rfc, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "AdaBoost":
        tuned_parameters = [{'random_state':[1,2,4,10,15], 'n_estimators': [10, 20, 50, 100, 200, 400],
                             'algorithm': ['SAMME', 'SAMME.R'], 'learning_rate': [1.0, 1.5, 2.0, 2.5]}]
        abc=AdaBoostClassifier()
        clf = GridSearchCV(abc, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "GBC":
        tuned_parameters = [{'max_depth':[1,2,4,10,15], 'n_estimators': [10, 20, 50, 100, 200, 400],
                             'loss': ['deviance'], 'learning_rate': [1.0, 1.5, 2.0, 2.5]}]
        gbc=GradientBoostingClassifier()
        clf = GridSearchCV(gbc, tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif algo == "XGBoost":
        tuned_parameters = [{'n_estimators': [10, 20, 50, 100, 200, 400],
                             'booster': ['gbtree', 'dart', 'gblinear'], 'learning_rate': [0.1, 1.0, 1.5, 2.0, 2.5],
                             'max_delta_step': [1, 5, 6, 10], 'seed': [None, 5]}]
        gbc=XGBClassifier()
        clf = GridSearchCV(gbc, tuned_parameters, cv=5,
                           scoring='%s' % score)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_pred))

    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
