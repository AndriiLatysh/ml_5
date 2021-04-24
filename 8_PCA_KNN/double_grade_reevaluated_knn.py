import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import sklearn.neighbors as sk_neighbours
import double_grade_utility


qualifies_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

for z in range(1, 10, 2):
    print(f"{z} neighbours:")

    double_grade_knn_model = sk_neighbours.KNeighborsClassifier(n_neighbors=z)
    cv_double_grade_knn_prediction = sk_model_selection.cross_val_predict(double_grade_knn_model, X, y, cv=4)

    cv_confusion_matrix = sk_metrics.confusion_matrix(y, cv_double_grade_knn_prediction)
    print(cv_confusion_matrix)

double_grade_knn_model = sk_neighbours.KNeighborsClassifier(n_neighbors=3)
double_grade_knn_model.fit(X, y)

double_grade_utility.plot_model(double_grade_knn_model, qualifies_double_grade)

plt.show()
