# %% carga de las librerías
import numpy as np
import pandas as pd
import statsmodels.api as sm            # regresion lineal
import statsmodels.formula.api as smf   # regresion lineal
import matplotlib.pyplot as plt         # visualizacion de datos
import seaborn as sns                   # visualizacion de datos
import yellowbrick.regressor as yb      # visualizacion de datos
import recursos.correlaciones as corr   # desarrollo propio


# %%: carga de los datos
ds2015 = pd.read_csv('datasets\\encuesta_felicidad_2015.csv', encoding='utf-8')
ds2016 = pd.read_csv('datasets\\encuesta_felicidad_2016.csv', encoding='utf-8')
ds2017 = pd.read_csv('datasets\\encuesta_felicidad_2017.csv', encoding='utf-8')


# %%: explorar la estructura de los datos
ds2015.info()


# %%: explorar la estructura de los datos
ds2015.head()


# %%: explorar la estructura de los datos
ds2015.Region.unique()


# %%: analizando el error porcentual
print("Promedio Felicidad %.4f" % np.average(ds2015.PuntajeFelicidad))
print("Promedio ErrorStd Absoluto %.4f" % np.average(ds2015.ErrorStd))
print("Promedio ErrorStd Relativo %.4f %%" % np.average(ds2015.ErrorStd / ds2015.PuntajeFelicidad * 100))


# %%: visualizar mediante histogramas
plt.figure(figsize=(10,7))
plt.xticks(rotation=90)

plt.title("Ocurrencias por Region")
plt.hist(ds2015.Region, bins=10)
plt.show()

plt.title("Ocurrencias por PuntajeFelicidad")
plt.hist(ds2015.PuntajeFelicidad, bins=10)
plt.show()

plt.title("Ocurrencias por Economía")
plt.hist(ds2015.Economia, bins=10)
plt.show()


# %%: distribución normal
total = len(ds2015)
partial = len(ds2015[(ds2015.PuntajeFelicidad >= 4) & (ds2015.PuntajeFelicidad < 6)])
print("Ocurrencias entre 4 a 6: %.2f %%" % (partial / total * 100))


# %%: diagrama de pares (posibles correlaciones)
vars=['PuntajeFelicidad','Economia','Familia','ExpectVida','Libertad','Instituciones','Generosidad','Justicia']
sns.pairplot(ds2015, vars=vars)
plt.show()


# %%: diagrama de dispersión
dsScatter = pd.DataFrame(ds2015, columns=['PuntajeFelicidad','Economia'])

# normalizar los datos (rango de 0 a 1)
dsScatter.PuntajeFelicidad = dsScatter.PuntajeFelicidad / max(dsScatter.PuntajeFelicidad)
dsScatter.Economia = dsScatter.Economia / max(dsScatter.Economia)

# diagrama de dispersión (posibles correlaciones)
plt.title('Economía vs Felicidad')
plt.xlabel('Economia')
plt.ylabel('Felicidad')
plt.scatter(dsScatter.Economia, dsScatter.PuntajeFelicidad, color='black')
plt.show()


# %%: cálculo de correlaciones (funcion propia)
dsCorr = pd.DataFrame(ds2015, columns=['PuntajeFelicidad','Economia','Familia','ExpectVida','Libertad','Instituciones','Generosidad','Justicia'])
corr.correlaciones(dsCorr, "PuntajeFelicidad")


# %%: cálculo de correlaciones
EconomiaFelicidad = np.corrcoef(ds2015.Economia, ds2015.PuntajeFelicidad)[0][1]
FamiliaFelicidad = np.corrcoef(ds2015.Familia, ds2015.PuntajeFelicidad)[0][1]
VidaFelicidad = np.corrcoef(ds2015.ExpectVida, ds2015.PuntajeFelicidad)[0][1]

print("Economia a Felicidad    %.2f" % EconomiaFelicidad)
print("Familia a Felicidad     %.2f" % FamiliaFelicidad)
print("ExpectVida a Felicidad  %.2f" % VidaFelicidad)


# %%: valores atípicos: boxplot
dsBoxPlot = pd.DataFrame(ds2015, columns=['PuntajeFelicidad', 'Economia', 'Familia', 'ExpectVida', 'Libertad', 'Instituciones', 'Generosidad', 'Justicia'])

# normalizar los datos (rango de 0 a 1)
dsBoxPlot.PuntajeFelicidad = dsBoxPlot.PuntajeFelicidad / max(dsBoxPlot.PuntajeFelicidad)
dsBoxPlot.Economia = dsBoxPlot.Economia / max(dsBoxPlot.Economia)
dsBoxPlot.Familia = dsBoxPlot.Familia / max(dsBoxPlot.Familia)
dsBoxPlot.ExpectVida = dsBoxPlot.ExpectVida / max(dsBoxPlot.ExpectVida)
dsBoxPlot.Libertad = dsBoxPlot.Libertad / max(dsBoxPlot.Libertad)
dsBoxPlot.Instituciones = dsBoxPlot.Instituciones / max(dsBoxPlot.Instituciones)
dsBoxPlot.Generosidad = dsBoxPlot.Generosidad / max(dsBoxPlot.Generosidad)
dsBoxPlot.Justicia = dsBoxPlot.Justicia / max(dsBoxPlot.Justicia)

plt.figure(figsize=(10,6))
labels = ['Felicidad', 'Economia', 'Familia', 'ExpectVida', 'Libertad', 'Instituciones', 'Generosidad', 'Justicia']

# diagrama de cajas
dsBoxPlot = [dsBoxPlot.PuntajeFelicidad, dsBoxPlot.Economia, dsBoxPlot.Familia, dsBoxPlot.ExpectVida, dsBoxPlot.Libertad, dsBoxPlot.Instituciones, dsBoxPlot.Generosidad, dsBoxPlot.Justicia]
plt.boxplot(dsBoxPlot, labels=labels)
plt.show()


# %%: valores atípicos: gráfico violín
plt.figure(figsize=(6,5))
plt.title("Instituciones")
sns.violinplot(ds2015.Instituciones, inner="points")
plt.show()
plt.figure(figsize=(6,5))
plt.title("Economia")
sns.violinplot(ds2015.Economia, inner="points")
plt.show()


# %%: dividir los datos previo a regresión

# calculo de cantidades 70/30
totalCount = ds2015.shape[0]         # 158 observaciones
trainCount = int(totalCount * 0.7)   # 70% para entrenamiento
testCount = totalCount - trainCount  # 30% para verificación

# desordenar aleatoriamente
dsAux = ds2015.sample(frac=1).reset_index(drop=True)

# separar en conjunto de Training y conjunto de Test
dsTrain = dsAux.iloc[:trainCount]   # desde el inicio del array hasta trainCount
dsTest = dsAux.iloc[-testCount:]    # desde el final del array en reversa


# %%: regresión lineal por 'mínimos cuadrados ordinarios'
y = dsTrain.PuntajeFelicidad
x1 = dsTrain.Economia
x2 = dsTrain.Familia
x3 = dsTrain.ExpectVida

# ols: ordinary least squares
model = smf.ols('y ~ x1 + x2 + x3', data=dsTrain)
results = model.fit()


# %%: summary()
print('-> Summary:')
print(results.summary())
print()


# %%: params()
print('-> Modelo.params:')
print(results.params)
print()
print('-> Modelo.mse_resid:')
print(results.mse_resid)
print()
print('-> La ecuación resultante es:')
textY = ('%.2f +' % results.params[0])
textX1 = ('%.2f X1 +' % results.params[1])
textX2 = ('%.2f X2 +' % results.params[2])
textX3 = ('%.2f X3 +' % results.params[3])
textErr = ('%.2f' % results.mse_resid)
print('f(x) = B0 + B1*Economía + B2*Familia + B3*ExpectVida + Err')
print('f(x) =', textY, textX1, textX2, textX3, textErr)
print()


# %%: predict()

# predecir valores para el dataframe de test
dicTestEquiv = dict(x1=dsTest.Economia, x2=dsTest.Familia, x3=dsTest.ExpectVida)
lsPredictedValues = results.predict(exog=dicTestEquiv)

# agregar como nueva columna
dsTest['PredictedFelicidad'] = lsPredictedValues


# %%: valores reales vs predicciones

# Train dataframe + Test dataframe
plt.figure(figsize=(11,5))
plt.title("Dataframes: Train + Test")
plt.scatter(dsTrain.Economia, dsTrain.PuntajeFelicidad, label="Training")
plt.scatter(dsTest.Economia, dsTest.PuntajeFelicidad, label="Testing")
plt.legend(loc="upper left")
plt.show()

# Test dataframe: valores predichos vs valores reales
plt.figure(figsize=(11,5))
plt.title("Valores predichos vs Valores reales")
plt.scatter(dsTest.Economia, dsTest.PuntajeFelicidad, color='green', label="Real")
plt.scatter(dsTest.Economia, dsTest.PredictedFelicidad, color='red', label="Predicho")
plt.legend(loc="upper left")
plt.show()


# %%: residuos
ybTrainX = dsTrain.Economia.values.reshape(trainCount, 1)
ybTrainY = dsTrain.PuntajeFelicidad.values.reshape(trainCount, 1)
ybTestX = dsTest.Economia.values.reshape(testCount, 1)
ybTestY = dsTest.PuntajeFelicidad.values.reshape(testCount, 1)
plt.figure(figsize=(10,7))
fig = yb.ResidualsPlot(yb.LinearRegression())
fig.fit(ybTrainX, ybTrainY) # los puntos en azul
fig.score(ybTestX, ybTestY) # los puntos en verde
fig.show()


# %%: QQ-plot
fig = sm.qqplot(results.resid, fit=True, line='45')
fig.show()
