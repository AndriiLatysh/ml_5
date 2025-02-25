import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree


def convert_to_numeric_values(df):
    converted_df = df.copy()
    converted_df = converted_df.replace({"history": {"bad": 0, "fair": 1, "excellent": 2},
                                         "income": {"low": 0, "high": 1},
                                         "risk": {"low": 0, "high": 1}})
    return converted_df


plt.figure(figsize=(12, 8))
loan_df = pd.read_csv("data/loans.csv")
print(loan_df)

numeric_loan_df = convert_to_numeric_values(loan_df)
print(numeric_loan_df)

feature_names = loan_df.columns.values[:-1]
X = numeric_loan_df[feature_names]
y = numeric_loan_df[["risk"]]

loan_decision_tree = sk_tree.DecisionTreeClassifier(criterion="entropy")

loan_decision_tree.fit(X, y)

sk_tree.plot_tree(loan_decision_tree, feature_names=feature_names, class_names=["low", "high"], filled=True, rounded=True)

plt.show()
