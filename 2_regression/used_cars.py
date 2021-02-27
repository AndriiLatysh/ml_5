import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear


used_cars_df = pd.read_csv("data/true_car_listings.csv")
print(len(used_cars_df))

print(used_cars_df[["Year"]].max())
used_cars_df[["Age"]] = used_cars_df[["Year"]].max() - used_cars_df[["Year"]]

print(used_cars_df.head())

model_list = used_cars_df[["Model", "Vin"]].groupby("Model").count().sort_values(by="Vin", ascending=False)
print(model_list.head(10))

selected_model_df = used_cars_df[used_cars_df["Model"] == "Civic"]
print(len(selected_model_df))

print(selected_model_df)

plt.scatter(selected_model_df[["Age"]], selected_model_df[["Price"]])

price_by_age_regression = sk_linear.LinearRegression()
price_by_age_regression.fit(selected_model_df[["Age"]], selected_model_df[["Price"]])

print(price_by_age_regression.coef_, price_by_age_regression.intercept_)

age_range = [[selected_model_df["Age"].min()], [selected_model_df[["Age"]].max()]]
predicted_price_by_age = price_by_age_regression.predict(age_range)

plt.plot(age_range, predicted_price_by_age)

plt.show()
