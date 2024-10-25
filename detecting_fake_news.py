import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Datos
df = pd.read_csv('D:\\DataFlair\\news.csv')
print("Shape: {}".format(*df.shape))
print(df.head())

labels = df.label
print(labels.head())
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Inizializar el TfIDf Vectorizador (Tf: Frecuencia de términos, IDf: Frecuencia de documento inversa)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Obtenemos las matrices de propiedades Tf e IDf para entrenamiento y test
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initializar el PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
# Predecir en el conjunto de test y calcular precisión
y_pred = pac.predict(tfidf_test)
precision = accuracy_score(y_test,y_pred)
print(f'Precisión: {round(precision*100,2)}%')

# Matriz de confusión
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
