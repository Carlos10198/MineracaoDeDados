import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

df_original = pd.read_csv("data/data.csv")

df_original['MSRP'] = pd.to_numeric(df_original['MSRP'], errors='coerce')

print("Colunas disponíveis:", df_original.columns)

total_original = len(df_original)
com_nan_geral = df_original.isnull().any(axis=1).sum()
sem_nan_geral = total_original - com_nan_geral

colunas_utilizadas = [
    'Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Vehicle Size',
    'Vehicle Style', 'highway MPG', 'city mpg'
]

colunas_com_msrp = colunas_utilizadas + ['MSRP']

df_filtrado_com_msrp = df_original[colunas_com_msrp].copy()

com_nan_final = df_filtrado_com_msrp.isnull().any(axis=1).sum()
sem_nan_final = len(df_filtrado_com_msrp) - com_nan_final

print(f"\nColunas utilizadas na análise (com MSRP): {len(colunas_com_msrp)} colunas")
print(f"Registros com dados faltantes (nas colunas usadas): {com_nan_final}")
print(f"Registros utilizados após limpeza (linhas completas nas colunas usadas): {sem_nan_final}")

df = df_filtrado_com_msrp.dropna().copy()

mediana = df['MSRP'].median()

df['categoria_preco'] = ['caro' if x > mediana else 'barato' for x in df['MSRP']]

df = df.drop(columns=['MSRP'])

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('categoria_preco', axis=1)
y = df['categoria_preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

pred_classes = label_encoders['categoria_preco'].inverse_transform(y_pred)

unique, counts = np.unique(pred_classes, return_counts=True)
contagem_classes = dict(zip(unique, counts))

print("\nQuantidade de carros classificados pelo modelo:")
for classe, qtd in contagem_classes.items():
    print(f"{classe}: {qtd}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.savefig("image/matriz_confusao.png")
plt.show()