import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics


qualifies_single_grade_df = pd.read_csv("data/single_grade.csv")
qualifies_single_grade_df.sort_values(by=["grade", "qualifies"], inplace=True)

X = qualifies_single_grade_df[["grade"]]
y = qualifies_single_grade_df["qualifies"]

plt.scatter(X, y)

qualification_model = sk_linear.LogisticRegression()
qualification_model.fit(X, y)

modeled_qualification = qualification_model.predict(X)
modeled_qualification_probability = qualification_model.predict_proba(X)[:, 1]

qualifies_single_grade_df["modeled probability"] = modeled_qualification_probability

print(qualifies_single_grade_df)

plt.plot(X, modeled_qualification, color="k")
plt.plot(X, modeled_qualification_probability, color="g")

confusion_matrix = sk_metrics.confusion_matrix(y, modeled_qualification)
print(confusion_matrix)

print("Accuracy:", sk_metrics.accuracy_score(y, modeled_qualification))
print("Error rate:", 1 - sk_metrics.accuracy_score(y, modeled_qualification))
print("Precision:", sk_metrics.precision_score(y, modeled_qualification))
print("Recall:", sk_metrics.recall_score(y, modeled_qualification))

plt.show()
