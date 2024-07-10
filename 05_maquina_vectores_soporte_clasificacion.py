import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def get_dataset(filename):
    if not os.path.isfile(filename):
        return None
    return pd.read_csv('HTTPS-clf-dataset.csv', delimiter=',', encoding='utf-8-sig')


# Llama a la función con el nombre de tu archivo
all_data = get_dataset("HTTPS-clf-dataset.csv")
#print(all_data.columns)



X = all_data.drop(columns=['TYPE','DBI_BRST_BYTES','DBI_BRST_PACKETS','PKT_LENGTHS','PPI_PKT_DIRECTIONS','PKT_TIMES','DBI_BRST_TIME_START',
                            'DBI_BRST_TIME_STOP','DBI_BRST_DURATION','DBI_BRST_INTERVALS','TIME_INTERVALS']).iloc[:,1:]
print(X)


le = LabelEncoder()
y = le.fit_transform(all_data['TYPE']) # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





vectores=SVC(kernel='rbf')
print("Inicio del Entrenamiento")
vectores.fit(X_train,y_train)



print("Inicio de prediccion")
y_pred=vectores.predict(X_test)



print("Matriz de confusion: ")
print(confusion_matrix(y_test,y_pred))



print("[#] Presicion del Modelo: ")
print(precision_score(y_test,y_pred,average=None))
print(precision_score(y_test,y_pred,average='micro'))








'''
############################################
plt.scatter(X_train['BYTES'], y_train)
plt.xlabel('BYTES')
plt.ylabel('TIPO DE TRAFICO')
plt.title('Relación entre BYTES y TIPO DE TRAFICO')
plt.show()


corr_matrix = X_train.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Matriz de correlación')
plt.show()



plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs. Valores reales')
plt.show()
'''