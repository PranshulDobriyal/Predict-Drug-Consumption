import numpy as np
import pandas as pd
from feature_selection import select_features
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import matplotlib.pyplot as plt
from sklearn import metrics
import os
idx_to_name = {
    0: "Alcohol",
    1: "Amphet",
    2: "Amyl",
    3: "Benzos",
    4: "Caff",
    5: "Cannabis",
    6: "Chock"
}

def br():
    print("#"*85)

def plot_confusion_matrices(path, confusion, clf):
    for i in range(len(confusion)):
        ConfusionMatrixDisplay(confusion_matrix=confusion[i][0], display_labels=clf[i].classes_).plot()
        train_path = f"{path}/Train"
        if not os.path.isdir(train_path):
            os.makedirs(train_path)
        plt.savefig(f"{train_path}/{idx_to_name[i]}.jpg")
        ConfusionMatrixDisplay(confusion_matrix=confusion[i][1], display_labels=clf[i].classes_).plot()
        test_path = f"{path}/Test"
        if not os.path.isdir(test_path):
            os.makedirs(test_path)
        plt.savefig(f"{train_path}/{idx_to_name[i]}.jpg")
        plt.savefig(f"{test_path}/{idx_to_name[i]}.jpg")

def plot_roc(path, roc):
    path = f"{path}/roc_curves/"
    for i in range(len(roc)):
        if not os.path.isdir(path):
            os.makedirs(path)

        #Extract data from roc variable
        fpr, tpr, thresh = roc[i]
        
        #Generate and save plot
        plt.figure()
        plt.plot(fpr, tpr, lw = 2)
        plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(f"{path}/{idx_to_name[i]}")

def run_classifier(classifier, input_columns, output_columns):
    clf = []
    accuracy = []
    confusion = []
    roc = []
    for i in selected_output_columns.columns:
        
        #Split the data into Train (67%) and Test (33%) Set
        X_train, X_test, y_train, y_test = train_test_split(input_columns,\
            output_columns[i].astype(int), test_size = 0.33, random_state = 5)
        
        #Train the Decision Tree Classifier
        
        classifier = classifier.fit(X_train, y_train.astype(int))

        #Calculate Train and Test Accuracy
        train_acc = classifier.score(X_train, y_train)
        test_acc = classifier.score(X_test, y_test)

        #Generate Confusion Matrices
        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)
        c_train = confusion_matrix(y_train, y_pred_train)
        c_test = confusion_matrix(y_test, y_pred_test)
        confusion.append([c_train, c_test])

        #Generate metrics required for ROC
        scores = classifier.predict_proba(X_test)
        fpr, tpr, thresh = metrics.roc_curve(y_test.to_numpy(), scores[:, 1])
        roc.append([
             fpr,
             tpr,
             thresh
        ])
        
        precision = precision_score(y_test, y_pred_test, average='binary')
        recall = recall_score(y_test, y_pred_test, average='binary')
        accuracy.append([train_acc, test_acc, precision, recall])

        clf.append(classifier)

    return {
        "Classifier": clf,
        "Confusion": confusion,
        "Accuracy": accuracy,
        "ROC": roc
    }


#Load the data
data = pd.read_csv("data/drug_consumption.data")
input_variables = data.iloc[:, 0:13]
output_variables = data.iloc[:, 13:]

#Convert C1 (CL0) and C2 (CL1) to non-user (0) and all other classes to user (1)
for column in output_variables.columns:
    output_variables.loc[(output_variables[column]  == "CL1") | (output_variables[column]  == "CL0")  , column] = 0
    output_variables.loc[output_variables[column] != 0, column] = 1


#Feature Selection
selected_input_columns = select_features(input_variables, mode = "VarianceThreshold")
selected_output_columns = select_features(output_variables)

classifiers = [
    [tree.DecisionTreeClassifier(min_samples_leaf=2), "decision_tree"],
    [RandomForestClassifier(max_depth=5, random_state=0), "random_forest"],
    [SVC(probability=True), "svm"],
    [KNeighborsClassifier(n_neighbors=3), "knn"]
    ]

for clas, name in classifiers:
    br()
    print({name})
    br()
    dt = run_classifier(clas, selected_input_columns, selected_output_columns)
    path = f"visual_data/{name}"
    plot_confusion_matrices(path, dt["Confusion"], dt["Classifier"])
    plot_roc(path, dt["ROC"])

    for i in range(len(dt["Accuracy"])):
        acc = dt["Accuracy"][i]
        br()
        print(f"{idx_to_name[i]} -- \n\n")
        print(f"\nTrain Accuracy: {acc[0]} \n")
        print(f"Test Accuracy: {acc[1]}")
        print(f"\nPrecision: {acc[2]}")
        print(f"\nRecall: {acc[3]}")