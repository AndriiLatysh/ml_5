import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree
import sklearn.ensemble as sk_ensemble
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics


diabetes_df = pd.read_csv("data/pima-indians-diabetes.csv")
column_names = diabetes_df.columns.values

X = diabetes_df[column_names[:-1]]
y = diabetes_df[column_names[-1]]

# print(len(diabetes_df))

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y)

print("Random forest:")

diabetes_forest_model = sk_ensemble.AdaBoostClassifier()

diabetes_forest_model.fit(X_train, y_train)
forest_y_predict = diabetes_forest_model.predict(X_test)

print("Accuracy:", sk_metrics.accuracy_score(y_test, forest_y_predict))

forest_confusion_matrix = sk_metrics.confusion_matrix(y_test, forest_y_predict)
print(forest_confusion_matrix)