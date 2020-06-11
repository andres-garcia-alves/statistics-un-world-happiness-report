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
dsAlfabetiz = dsAlfabetiz.drop(['Operacion','Fuente','Anio','Variacion'], axis=1)

cols = {'Adultos':'AlfabetizGral','Mujeres':'AlfabetizMuj','Hombres':'AlfabetizHom','Jovenes':'AlfabetizJov'}
dsAlfabetiz = dsAlfabetiz.rename(columns=cols)

# combinar los datos
dsCombinado = pd.merge(left=dsFelicidad, right=dsAlfabetiz, left_on='Pais', right_on='Pais')
dsCombinado.dropna(inplace = True)

# normalizar los datos (rango de 0 a 1)
dsCombinado.PuntajeFelicidad = dsCombinado.PuntajeFelicidad / max(dsCombinado.PuntajeFelicidad)
dsCombinado.head()


# %%: visualizar mediante histogramas
plt.figure(figsize=(10,7))
plt.xticks(rotation=90)

plt.title('Porcentajes de Alfabetización')
plt.hist(dsCombinado.AlfabetizGral, bins=10)
plt.show()


# %%: valores atípicos: boxplot
series = [dsCombinado.PuntajeFelicidad, dsCombinado.AlfabetizGral]
labels = ['Felicidad', 'Alfabetizacion']

plt.figure(figsize=(10,6))
plt.boxplot(series, labels=labels)
plt.show()


# %%: valores atípicos: gráfico violín
plt.figure(figsize=(10,6))
plt.title("Alfabetizacion")
sns.violinplot(dsCombinado.AlfabetizGral, inner="points")
plt.show()


# %%: diagrama de pares (posibles correlaciones)
vars=['PuntajeFelicidad','Economia','Familia','ExpectVida','AlfabetizGral']
sns.pairplot(dsCombinado, vars=vars)
plt.show()


# %%: diagrama de dispersión (posibles correlaciones)
plt.figure(figsize=(8,8))
plt.title('Alfabetización vs Felicidad')
plt.xlabel('Alfabetiz')
plt.ylabel('Felicidad')
plt.scatter(dsCombinado.AlfabetizGral, dsCombinado.PuntajeFelicidad, color='black')
plt.show()


# %%: cálculo de correlaciones
coefEconomia = np.corrcoef(dsCombinado.AlfabetizGral, dsCombinado.Economia)[0][1]
coefFamilia = np.corrcoef(dsCombinado.AlfabetizGral, dsCombinado.Familia)[0][1]
coefExpectVida = np.corrcoef(dsCombinado.AlfabetizGral, dsCombinado.ExpectVida)[0][1]
coefLibertad = np.corrcoef(dsCombinado.AlfabetizGral, dsCombinado.Libertad)[0][1]
coefInstituciones = np.corrcoef(dsCombinado.AlfabetizGral, dsCombinado.Instituciones)[0][1]
coefGenerosidad = np.corrcoef(dsCombinado.AlfabetizGral, dsCombinado.Generosidad)[0][1]
coefJusticia = np.corrcoef(dsCombinado.AlfabetizGral, dsCombinado.Justicia)[0][1]

print("Correlaciones de Alfabetización:")
print(" vs Economia      %.2f" % coefEconomia)
print(" vs Familia       %.2f" % coefFamilia)
print(" vs ExpectVida    %.2f" % coefExpectVida)
print(" vs Libertad      %.2f" % coefLibertad)
print(" vs Instituciones %.2f" % coefInstituciones)
print(" vs Generosidad   %.2f" % coefGenerosidad)
print(" vs Justicia      %.2f" % coefJusticia)


# %%: diagramas de la regresión: lm (lineal model)
g = sns.lmplot("AlfabetizGral", "PuntajeFelicidad", dsCombinado, height=6)
g.set_axis_labels("Alfabetización", "Felicidad")
plt.show()


# %%: diagramas de la regresión: joint, subtipo 'reg'
g = sns.jointplot(dsCombinado.AlfabetizGral, dsCombinado.PuntajeFelicidad, kind="reg", height=8, ratio=3, color="r")
g.set_axis_labels("Alfabetización", "Felicidad")
plt.show()


#%%: gráfico de torta: factores de felicidad en la Argentina
dsArgentina = dsFelicidad[dsFelicidad.Pais == 'Argentina']
dsArgentina['Otros'] = dsArgentina.PuntajeFelicidad - (dsArgentina.Economia + dsArgentina.Familia + dsArgentina.ExpectVida + dsArgentina.Justicia)
dsPieChart = pd.DataFrame(dsArgentina, columns=["Economia","Familia","ExpectVida","Justicia","Otros"])

# gráfico de torta
plt.figure(figsize=(7,7))
plt.title('Factores de felicidad en Argentina', fontsize=16)
plt.pie(dsPieChart.values, labels=dsPieChart.columns, autopct='%1.1f%%')
plt.show()


# %%: deterioro  vs mejora en alfabetización
dsIgual = dsCombinado[dsCombinado.AlfabetizJov == dsCombinado.AlfabetizGral] # 27
dsMejora = dsCombinado[dsCombinado.AlfabetizJov > dsCombinado.AlfabetizGral] # 125
dsDeterioro = dsCombinado[dsCombinado.AlfabetizJov < dsCombinado.AlfabetizGral] # 2

lsTotals = [len(dsIgual), len(dsMejora), len(dsDeterioro)]
labels = ['Igual','Mejora','Deterioro']
colors = ['Blue','Green','Red']

# gráfico de torta
plt.figure(figsize=(7,7))
plt.title('Evolución generacional de la alfabetización', fontsize=16)
plt.pie(lsTotals, labels=labels, colors=colors, autopct='%1.1f%%')
plt.show()


# %%: composición de la felicidad entre mejor y peor decil de alfabetización

# ordenar x alfabetización
dsDeciles = dsCombinado.sort_values("AlfabetizGral", ascending=False)

# asignar & separar por deciles
series = []
length = len(dsDeciles) * 0.1
for i in range(len(dsCombinado)): series.append(int(i / length) + 1)

dsDeciles['Decil'] = series
dsDeciles = dsDeciles[(dsDeciles.Decil == 1) | (dsDeciles.Decil == 10)]

# calcular promedios
lstEconomia = []; lstFamilia = []; lstExpectVida = []; lstLibertad = []
lstInstituciones = []; lstGenerosidad = []; lstJusticia = []

for number in [1, 10]:   
    dsDecil = dsDeciles[dsDeciles.Decil == number] # los datos del decil

    lstEconomia.append(sum(dsDecil.Economia) / len(dsDecil))
    lstFamilia.append(sum(dsDecil.Familia) / len(dsDecil))
    lstExpectVida.append(sum(dsDecil.ExpectVida) / len(dsDecil))
    lstLibertad.append(sum(dsDecil.Libertad) / len(dsDecil))
    lstInstituciones.append(sum(dsDecil.Instituciones) / len(dsDecil))
    lstGenerosidad.append(sum(dsDecil.Generosidad) / len(dsDecil))
    lstJusticia.append(sum(dsDecil.Justicia) / len(dsDecil))

# armar el nuevo dataframe
dsBarPlot = pd.DataFrame(index=['Decil 1','Decil 10'], data={ 
    'Economia':lstEconomia, 'Familia':lstFamilia, 'ExpectVida':lstExpectVida,
    'Libertad':lstLibertad, 'Instituciones':lstInstituciones,
    'Generosidad':lstGenerosidad, 'Justicia':lstJusticia })
dsBarPlot


# %%: gráfico de barras
plt.figure(figsize=(10,8))
plt.rcParams.update({'axes.titlesize': 'large'})
title = 'Composición de la felicidad entre mejor y peor decil de alfabetización'

dsBarPlot.plot(kind='bar', stacked=False, figsize=(12,6), rot=0, title=title, fontsize=12)
plt.show()


# %%: gráfico de barras apiladas
# f, ax = plt.subplots(figsize=(10,2))

# sns.set(style="whitegrid")
# sns.barplot(data=dsBarPlot, x='Familia', y=dsBarPlot.index, color='orange', orient='h', label='Familia')
# sns.barplot(data=dsBarPlot, x='Economia', y=dsBarPlot.index, color='blue', orient='h', label='Economia')
# sns.barplot(data=dsBarPlot, x='ExpectVida', y='Decil', color='green', orient='h', label='ExpectVida')
# sns.barplot(data=dsBarPlot, x='Libertad', y='Decil', color='red', orient='h', label='Libertad')
# sns.barplot(data=dsBarPlot, x='Instituciones', y='Decil', color='yellow', orient='h', label='Instituciones')
# sns.barplot(data=dsBarPlot, x='Generosidad', y='Decil', color='cyan', orient='h', label='Generosidad')
# sns.barplot(data=dsBarPlot, x='Justicia', y='Decil', color='black', orient='h', label='Justicia')

# ax.legend(ncol=4, bbox_to_anchor=(0.85,-0.4))
# ax.set(xlabel='Componentes', ylabel='Deciles')
# ax.set_title('Composición de la felicidad entre mejor y peor decil de alfabetización', fontsize=16, pad=20)
# plt.show()
