import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection


qualifies_double_grade = pd.read_csv("data/double_grade.csv")

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

qualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 1]
unqualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 0]

plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")

plt.xlabel("Technical grade")
plt.ylabel("English grade")

number_of_folds = 4

cv_qualification_model = sk_linear.LogisticRegression()
cv_model_quality = sk_model_selection.cross_val_score(cv_qualification_model, X, y, cv=number_of_folds, scoring="accuracy")
print(cv_model_quality)

cv_prediction = sk_model_selection.cross_val_predict(cv_qualification_model, X, y, cv=number_of_folds)
cv_confusion_matrix = sk_metrics.confusion_matrix(y, cv_prediction)
print(cv_confusion_matrix)

qualification_model = sk_linear.LogisticRegression()
qualification_model.fit(X, y)

modeled_qualification_probability = qualification_model.predict_proba(X)[:, 1]
qualifies_double_grade["modeled probability"] = modeled_qualification_probability

pd.set_option("display.max_rows", None)
print(qualifies_double_grade.sort_values(by="modeled probability"))

print(qualification_model.coef_)
print(qualification_model.intercept_)

k1, k2 = qualification_model.coef_.flatten()
b = qualification_model.intercept_[0]

x_boundary = [qualifies_double_grade["technical_grade"].min(), qualifies_double_grade["technical_grade"].max()]
y_boundary = [-(k1 * x + b) / k2 for x in x_boundary]

plt.plot(x_boundary, y_boundary, color="b")

plt.clf()

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")

false_positive_rate, true_positive_rate, thresholds = sk_metrics.roc_curve(y, modeled_qualification_probability)
plt.plot(false_positive_rate, true_positive_rate)

roc_auc = sk_metrics.roc_auc_score(y, modeled_qualification_probability)
print("Area under curve:", roc_auc)

plt.show()
