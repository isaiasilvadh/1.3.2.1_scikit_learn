```python
"""
Ejemplo tomado de 
Pineda Pertuz, C. (2022). Aprendizaje automático y profundo en Python: una mirada hacia la inteligencia artificial: 
(1 ed.). Madrid, RA-MA Editorial. Recuperado de https://elibro.net/es/ereader/bibliotecautpl/230579?page=73.
"""
```


```python
import pandas as pd
# se importa y se carga el dataset
from sklearn.datasets import load_breast_cancer 
dataset = load_breast_cancer()
# se convierte a un dataframe de pandas

```


```python
# se convierte a un dataframe de pandas 
df = pd.DataFrame(dataset.data, columns=dataset.feature_names) 
df['tipo'] = dataset.target[df.index] 
# se obtienen los valores para las variables X e y 
X = df.iloc[:,:-1] 
y = df['tipo'].values
```


```python
# se separan los datos en conjuntos de prueba y entrenamiento 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # se crea una instancia de la clase LogisticRegression  
from sklearn.linear_model import LogisticRegression 
reg = LogisticRegression(max_iter=10000) 
# se entrena el modelo con lo datos de entrenamiento 
reg.fit(X_train,y_train) 
from sklearn.metrics import accuracy_score
print("Exactitud {:%.2f }"%(accuracy_score(y_test, reg.predict(X_test))))

```

    Exactitud {:0.94 }

