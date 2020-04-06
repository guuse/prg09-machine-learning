import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

from machinelearningdata import Machine_Learning_Data

def extract_from_json_as_np_array(key, json_data):
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)

STUDENTNUMMER = "0912071"
data = Machine_Learning_Data(STUDENTNUMMER)

show_diff = True
start_test = False

classification_training = data.classification_training()
X = extract_from_json_as_np_array("x", classification_training)
Y = extract_from_json_as_np_array("y", classification_training)

## Leer de classificaties
x_2 = X[...,0]
y_2 = X[...,1]
plt.axis([min(x_2), max(x_2), min(y_2), max(y_2)])

## Train de data in een decision tree, maak de y_predict aan gebaseerd op de data
decisionTree = tree.DecisionTreeClassifier().fit(X, Y)
Y_predict = decisionTree.predict(X)

## Bereken de accuracy score gebaseerd op je Y_predict t.o.v daadwerkelijke Y
initialTreeScore = accuracy_score(Y, Y_predict)
print("Initial accuracy score (Decision Tree): " + str(float(initialTreeScore) * 100) + "%")

## vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt, 0 is paars & 1 is geel
plt.scatter(x_2, y_2, c = Y_predict, s = 10)
if show_diff:
    plt.show()

if start_test:
    classification_test = data.classification_test()
    X_test = extract_from_json_as_np_array("x", classification_test)

    ## Voorspelt nu echt de Y-waarden
    Z = decisionTree.predict(X_test)

    classification_test = data.classification_test(Z.tolist())
    print("Trained decision tree accuracy score (Decision Tree): " + str(float(classification_test) * 100) + "%")
