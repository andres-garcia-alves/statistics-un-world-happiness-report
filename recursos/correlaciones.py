# %%
import numpy as np
import pandas as pd
from sklearn import preprocessing

# %%
def correlaciones(data: pd.core.frame.DataFrame, colNameX: str, normalize: bool = True):
  
  print("Análisis de Correlaciones vs '{0}'".format(colNameX))

  for colNameY in data.columns:      
    if colNameY != colNameX:

      # par de columnas a procesar en la iteración actual
      x = data[colNameX]
      y = data[colNameY]

      # normalizar los datos?
      if normalize == True:
        x = preprocessing.normalize(np.array([x]))
        y = preprocessing.normalize(np.array([y]))

      # calculo de correlacion para la columna 'colNameX' vs 'colNameY'
      corr = np.corrcoef(x, y)
      
      # mostrar resultado
      resultColumnOne = colNameY.ljust(25)
      resultColumnTwo = corr[0][1]
      print("- {0} {1:.4f}".format(resultColumnOne, resultColumnTwo))

  print()
