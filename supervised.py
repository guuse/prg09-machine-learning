import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from machinelearningdata import Machine_Learning_Data

def extract_from_json_as_np_array(key, json_data):
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)

STUDENTNUMMER = "0912071"
data = Machine_Learning_Data(STUDENTNUMMER)




# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)


# TODO: leer de classificaties

# TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict

# TODO: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt


# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)

# TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.

Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie (test): " + str(classification_test))

