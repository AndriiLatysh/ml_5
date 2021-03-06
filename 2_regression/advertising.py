import pandas as pd
import numpy as np
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection


def model_to_string(model, labels, precision=4):
    model_str = "{} = ".format(labels[-1])
    for z in range(len(labels) - 1):
        model_str += "{} * {} + ".format(round(model.coef_.flatten()[z], precision), labels[z])
    model_str += "{}".format(round(model.intercept_[0]), precision)
    return model_str


def train_linear_model(X, y):
    linear_model = sk_linear.LinearRegression()
    linear_model.fit(X, y)
    return linear_model


def get_MSE(model, X, y_true):
    y_predicted = model.predict(X)
    MSE = sk_metrics.mean_squared_error(y_true, y_predicted)
    return MSE


advertising_data = pd.read_csv("data/advertising.csv", index_col=0)
print(advertising_data)

ad_data = advertising_data[["TV", "radio", "newspaper"]]
sales_data = advertising_data[["sales"]]

labels = advertising_data.columns.values

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(ad_data, sales_data, shuffle=True)

linear_regression = train_linear_model(X_train, y_train)

print("General model:")
print(model_to_string(linear_regression, labels))
print("Train MSE:", get_MSE(linear_regression, X_train, y_train))
print("Test MSE:", get_MSE(linear_regression, X_test, y_test))
print()

for z in range(0, 3):
    feature_name = labels[z]
    print("{} removed:".format(feature_name))

    print("Pearson correlation coefficient between {} and sales: {}".format(feature_name,
                                                                            np.corrcoef(ad_data[feature_name],
                                                                                        sales_data["sales"])[0][1]))

    X_train_2_features = X_train.drop(feature_name, axis=1)
    X_test_2_features = X_test.drop(feature_name, axis=1)
    labels_2_features = np.delete(labels, z)

    model_2_features = train_linear_model(X_train_2_features, y_train)
    print(model_to_string(model_2_features, labels_2_features))
    print("MSE:", get_MSE(model_2_features, X_test_2_features, y_test))
    print()
