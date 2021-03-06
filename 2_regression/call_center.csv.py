import pandas as pd
import numpy as np
import sklearn.linear_model as sk_linear
import matplotlib.pyplot as plt
import mlinsights.mlmodel as mi_models


plt.figure(figsize=(20, 8))

call_center_df = pd.read_csv("data/call_center.csv", parse_dates=["timestamp"])
# plt.plot(call_center_df[["timestamp"]], call_center_df[["calls"]])
# print(call_center_df.dtypes)

X = np.array([t.value / 1e18 for t in call_center_df["timestamp"]]).reshape(-1, 1)
y = call_center_df[["calls"]]

plt.plot(X, y, color="b")

border_values = [[X[0][0]], [X[-1][0]]]
print(border_values)

print("OLS (MSE):")

ols_model = sk_linear.LinearRegression()
ols_model.fit(X, y)

ols_trend = ols_model.predict(border_values)

plt.plot(border_values, ols_trend, color="r")

print("Slope:", ols_model.coef_[0][0])
print("Overall change:", ols_trend[1][0] - ols_trend[0][0])

print("LAD (MAE):")

y = call_center_df["calls"]

lad_model = mi_models.QuantileLinearRegression()
lad_model.fit(X, y)

lad_trend = lad_model.predict(border_values)

plt.plot(border_values, lad_trend, color="g")

print("Slope:", lad_model.coef_[0])
print("Overall change:", lad_trend[1] - lad_trend[0])

plt.show()
