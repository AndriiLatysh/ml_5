import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk_cluster
import sklearn.preprocessing as sk_preprocessing
import scipy.cluster.hierarchy as sp_cluster_hr


def set_printing_options():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)


set_printing_options()

clients = pd.read_csv("data/customer_online_closing_store.csv")

clients["return_rate"] = clients["items_returned"] / clients["items_purchased"]
clients["average_price"] = clients["total_spent"] / clients["items_purchased"]

X = clients[["average_price", "return_rate", "overall_rating"]]
print(X)

min_max_scaler = sk_preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
print(X)

linkage_method = "ward"
dendrogram = sp_cluster_hr.dendrogram(sp_cluster_hr.linkage(X, method=linkage_method))

agglomerative_model = sk_cluster.AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
agglomerative_model.fit(X)

clients["class"] = agglomerative_model.labels_
print(clients[["average_price", "return_rate", "overall_rating", "class"]])

client_pivot_table = clients.pivot_table(index="class",
                                         values=["average_price", "return_rate", "overall_rating", "customer_id"],
                                         aggfunc={"average_price": np.mean, "return_rate": np.mean,
                                                  "overall_rating": np.mean, "customer_id": len})

print(client_pivot_table)

plt.show()
