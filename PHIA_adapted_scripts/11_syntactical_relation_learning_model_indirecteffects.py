import os
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import pickle

#################################################################################################
## This script test multiple methods for feature based machine learning models that predict the
## truthfulness of a proposed relation between phrases in sentences based on the syntactical
## feautures of the sentence (as generated in 10_syntactical_relation_learning_preparation).
#################################################################################################

def ReportOnModel10CrossValid(cv, model, X, y):
    try:
        n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        n_scores = cross_val_score(model, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1, error_score='raise')
        print('F1 Score weighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        n_scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
        print('F1 Score true values: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        return mean(n_scores)
    except ValueError as e:
        print("Cross-validation error:", e)
        return 0

def ReportOnTotalModel(values, prediction):
    report = classification_report(values, prediction, target_names=['0', '1'])
    print(report)
    reportlist = report.split()
    totalF1_true = reportlist[(reportlist.index("1"))+3]
    print("Total Training F1 score for True: ", totalF1_true)
    return totalF1_true

# Set up paths
os.chdir(r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Indirect_Effects\final_kg")
csv = os.path.join(os.getcwd(), "possible_evidence_instances400.csv")
traindata = pd.read_csv(csv)

print(traindata.iloc[:,9:].head())  # Check columns
best_F1true_CV = 0

# Define dataset
y = traindata['Evidence_Truth']
X = traindata.iloc[:, 9:].apply(pd.to_numeric, errors='coerce').fillna(-100)  # Ensure all are numeric and handle NaNs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)

# Model configurations
model_configs = [
    ("Gradient_Boosting", GradientBoostingClassifier()),
    ("Random_Forest", RandomForestClassifier()),
    ("Multi_layer_Perceptron", MLPClassifier(solver='lbfgs', max_iter=3000, hidden_layer_sizes=(5, 2))),
    ("Support_vector_machines", svm.SVC())
]

# Train and evaluate models
for modeltype, model in model_configs:
    print(f"\n{modeltype}\n===============")
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    F1_true = ReportOnModel10CrossValid(cv, model, X, y)
    
    try:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        totalF1_true = ReportOnTotalModel(y_test.values, prediction)

        if F1_true > best_F1true_CV:
            bestmodeltype = modeltype
            bestmodel = model
            best_F1true_CV = F1_true
            best_F1true_total = totalF1_true

    except ValueError as e:
        print(f"Error in {modeltype} model fitting:", e)

# Save best model
print("The best model type is:", bestmodeltype)
print("10-fold CV F1 score for predicting true relations:", best_F1true_CV)
print("Total train data F1 score for predicting true relations:", best_F1true_total)

output_model_path = f'C:/Users/4209416/OneDrive - Universiteit Utrecht/Desktop/ETAIN/PHIA_framework/Automated-KG/newCrossRef/Indirect_Effects/syntactRelatModel_{bestmodeltype}_F1TrueCV{str(round(best_F1true_CV, 2)).replace(".", "_")}_F1TrueTot{best_F1true_total.replace(".", "_")}.pickle'
pickle.dump(bestmodel, open(output_model_path, 'wb'))

# Prediction on new data
pred_csv_path = os.path.join(os.getcwd(), "possible_evidence_instances4_total.csv")
preddata = pd.read_csv(pred_csv_path)
preddata.replace({'NaN': -100}, regex=True, inplace=True)
preddata.iloc[:, 9:] = preddata.iloc[:, 9:].apply(pd.to_numeric, errors='coerce').fillna(-100)  # Ensure numeric

predictions = []
for i in range(len(preddata)):
    try:
        predictions.append(model.predict([preddata.iloc[i, 9:].values]))  # Adjust to pass array correctly
        if i % 1000 == 0:
            print("Completed instance:", i)
    except NotFittedError as e:
        print("Prediction error at instance", i, ":", e)
        predictions.append(-1)  # Default value for failed predictions

preddata["Evidence_Truth"] = np.array(predictions).flatten()
preddata.to_csv("predicted_evidence_instances_all.csv", index=False)

# Save true relation predictions
true_evidence_instances = preddata[preddata["Evidence_Truth"] == 1].drop_duplicates()
print("Nr of true relation predictions:", len(true_evidence_instances))
print("Nr of unique true relation predictions:", len(true_evidence_instances))
true_evidence_instances.to_csv("predicted_evidence_instances_true.csv", index=False)
