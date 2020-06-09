# %% carga de las librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt         # visualizacion de datos
import seaborn as sns                   # visualizacion de datos


# %%: carga y limpieza de los datos

# carga de los datos
dsFelicidad = pd.read_csv('datasets\\encuesta_felicidad_2015.csv', encoding='utf-8')
dsAlfabetiz = pd.read_csv('datasets\\alfabetizacion_v04.csv', encoding='ISO-8859-1')

# eliminar filas y columnas innecesarias
dsFelicidad = dsFelicidad.drop(['ErrorStd','Libertad','Instituciones','Generosidad','Justicia'], axis=1)
dsAlfabetiz = dsAlfabetiz[dsAlfabetiz['Operacion'] != 'IGNORAR']
dsAlfabetiz = dsAlfabetiz.drop(['Operacion','Fuente','Anio','Mujeres','Hombres','Jovenes','Variacion'], axis=1)
dsAlfabetiz = dsAlfabetiz.rename(columns={'Adultos':'Alfabetiz'})

# combinar los datos
dsCombined = pd.merge(left=dsFelicidad, right=dsAlfabetiz, left_on='Pais', right_on='Pais')
dsCombined.dropna(inplace = True)

# normalizar los datos (rango de 0 a 1)
dsCombined.PuntajeFelicidad = dsCombined.PuntajeFelicidad / max(dsCombined.PuntajeFelicidad)
dsCombined.head()


# %%: visualizar mediante histogramas
plt.figure(figsize=(10,7))
plt.xticks(rotation=90)

plt.title("Porcentajes de Alfabetización")
plt.hist(dsCombined.Alfabetiz, bins=10)
plt.show()


# %%: valores atípicos: boxplot
series = [dsCombined.PuntajeFelicidad, dsCombined.Alfabetiz]
labels = ['Felicidad', 'Alfabetizacion']

plt.figure(figsize=(10,6))
plt.boxplot(series, labels=labels)
plt.show()


# %%: valores atípicos: gráfico violín
plt.figure(figsize=(10,6))
plt.title("Alfabetizacion")
sns.violinplot(dsCombined.Alfabetiz, inner="points")
plt.show()


# %%: diagrama de pares (posibles correlaciones)
vars=['PuntajeFelicidad','Economia','Familia','ExpectVida','Alfabetiz']
sns.pairplot(dsCombined, vars=vars)
plt.show()


# %%: diagrama de dispersión (posibles correlaciones)
plt.title('Alfabetización vs Felicidad')
plt.xlabel('Alfabetiz')
plt.ylabel('Felicidad')
plt.scatter(dsCombined.Alfabetiz, dsCombined.PuntajeFelicidad, color='black')
plt.show()


# %%: cálculo de correlaciones
AlfabetizFelicidad = np.corrcoef(dsCombined.Alfabetiz, dsCombined.PuntajeFelicidad)[0][1]
print("Alfabetización y Felicidad:")
print("Correlación    %.2f" % AlfabetizFelicidad)


# %%: diagramas de la regresión: lm (lineal model)
g = sns.lmplot("Alfabetiz", "PuntajeFelicidad", dsCombined, height=6)
g.set_axis_labels("Alfabetización", "Felicidad")
plt.show()


# %%: diagramas de la regresión: joint, subtipo 'reg'
g = sns.jointplot(dsCombined.Alfabetiz, dsCombined.PuntajeFelicidad, kind="reg", height=8, ratio=3, color="r")
g.set_axis_labels("Alfabetización", "Felicidad")
plt.show()


# %%
# equivalente a: 'SELECT PuntajeFelicidad, Economia GROUP BY Region'
# dsInstitucFelicidad = pd.pivot_table(ds2015, index='Region', values=["PuntajeFelicidad", "Economia"])
