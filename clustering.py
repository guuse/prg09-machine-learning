import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from machinelearningdata import Machine_Learning_Data

def extract_from_json_as_np_array(key, json_data):
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)

STUDENTNUMMER = "0912071"
data = Machine_Learning_Data(STUDENTNUMMER)

kmeans_training = data.clustering_training()
X = extract_from_json_as_np_array("x", kmeans_training)
x = X[...,0]
y = X[...,1]

kmeans_disabled = True
kmeans_clusters = 0

if kmeans_disabled:
    for i in range(len(x)):
        plt.plot(x[i], y[i], "k.")
else:
    kmeans = KMeans(n_clusters=kmeans_clusters)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    colors = ["r", "b", "g", "y", "m", "c"]

    for i in range(len(x)):
        plt.plot(x[i], y[i], "%s." % (colors[labels[i]]))

    for i in range(len(centroids)):
        Cx = centroids[i][...,0]
        Cy = centroids[i][...,1]
        plt.scatter(Cx, Cy, marker = "X", alpha = .5, s = 1000, c = colors[i], edgecolors = "face", linewidths = 2)

plt.axis([min(x), max(x), min(y), max(y)])
plt.show()
