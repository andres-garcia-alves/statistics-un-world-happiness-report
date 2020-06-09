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
dsFelicidad = dsFelicidad.drop(['ErrorStd'], axis=1)
dsAlfabetiz = dsAlfabetiz[dsAlfabetiz['Operacion'] != 'IGNORAR']
dsAlfabetiz = dsAlfabetiz.drop(['Operacion','Fuente','Anio','Mujeres','Hombres','Jovenes','Variacion'], axis=1)
dsAlfabetiz = dsAlfabetiz.rename(columns={'Adultos':'Alfabetiz'})

# combinar los datos
dsCombinado = pd.merge(left=dsFelicidad, right=dsAlfabetiz, left_on='Pais', right_on='Pais')
dsCombinado.dropna(inplace = True)

# normalizar los datos (rango de 0 a 1)
dsCombinado.PuntajeFelicidad = dsCombinado.PuntajeFelicidad / max(dsCombinado.PuntajeFelicidad)
dsCombinado.head(15)


# %%: visualizar mediante histogramas
plt.figure(figsize=(10,7))
plt.xticks(rotation=90)

plt.title("Porcentajes de Alfabetización")
plt.hist(dsCombinado.Alfabetiz, bins=10)
plt.show()


# %%: valores atípicos: boxplot
series = [dsCombinado.PuntajeFelicidad, dsCombinado.Alfabetiz]
labels = ['Felicidad', 'Alfabetizacion']

plt.figure(figsize=(10,6))
plt.boxplot(series, labels=labels)
plt.show()


# %%: valores atípicos: gráfico violín
plt.figure(figsize=(10,6))
plt.title("Alfabetizacion")
sns.violinplot(dsCombinado.Alfabetiz, inner="points")
plt.show()


# %%: diagrama de pares (posibles correlaciones)
vars=['PuntajeFelicidad','Economia','Familia','ExpectVida','Alfabetiz']
sns.pairplot(dsCombinado, vars=vars)
plt.show()


# %%: diagrama de dispersión (posibles correlaciones)
plt.title('Alfabetización vs Felicidad')
plt.xlabel('Alfabetiz')
plt.ylabel('Felicidad')
plt.scatter(dsCombinado.Alfabetiz, dsCombinado.PuntajeFelicidad, color='black')
plt.show()


# %%: cálculo de correlaciones
AlfabetizFelicidad = np.corrcoef(dsCombinado.Alfabetiz, dsCombinado.PuntajeFelicidad)[0][1]
print("Alfabetización y Felicidad:")
print("Correlación    %.2f" % AlfabetizFelicidad)


# %%: diagramas de la regresión: lm (lineal model)
g = sns.lmplot("Alfabetiz", "PuntajeFelicidad", dsCombinado, height=6)
g.set_axis_labels("Alfabetización", "Felicidad")
plt.show()


# %%: diagramas de la regresión: joint, subtipo 'reg'
g = sns.jointplot(dsCombinado.Alfabetiz, dsCombinado.PuntajeFelicidad, kind="reg", height=8, ratio=3, color="r")
g.set_axis_labels("Alfabetización", "Felicidad")
plt.show()


#%%: factores de felicidad en la Argentina
dsArgentina = dsFelicidad[dsFelicidad.Pais == 'Argentina']
dsArgentina['Otros'] = dsArgentina.PuntajeFelicidad - (dsArgentina.Economia + dsArgentina.Familia + dsArgentina.ExpectVida + dsArgentina.Justicia)
dsPivot = pd.pivot_table(dsArgentina, index='', values=["Economia","Familia","ExpectVida","Justicia","Otros"])

# gráfico de torta
plt.figure(figsize=(7,7))
plt.title('Factores de felicidad en Argentina', fontsize=15)
plt.pie(dsPivot.values, labels=dsPivot.columns, autopct='%1.1f%%')
plt.show()


# %%: promedio de felicidad por región          FALTA, CONT ACA !!!
regionList = list(dsCombinado.Region.unique())

promFelicidad = []
promEconomia = []
promFamilia = []
promExpectVida = []
promAlfabetiz = []

for regionItem in regionList:
    # seleccionar toda la region
    region = dsCombinado[dsCombinado.Region == regionItem]

    # calcular promedio
    promFelicidad.append(sum(region.PuntajeFelicidad) / len(region))
    promEconomia.append(sum(region.Economia) / len(region))
    promFamilia.append(sum(region.Familia) / len(region))
    promExpectVida.append(sum(region.ExpectVida) / len(region))
    promAlfabetiz.append(sum(region.Alfabetiz) / len(region))

