import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import sklearn.naive_bayes as sk_naive_bayes
import double_grade_utility


qualifies_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

sns.pairplot(qualifies_double_grade, hue="qualifies")

naive_bayes_model = sk_naive_bayes.GaussianNB()
cv_prediction = sk_model_selection.cross_val_predict(naive_bayes_model, X, y, cv=4)

confusion_matrix = sk_metrics.confusion_matrix(y, cv_prediction)
print(confusion_matrix)

naive_bayes_model.fit(X, y)

plt.figure()
double_grade_utility.plot_model(naive_bayes_model, qualifies_double_grade)

plt.show()
