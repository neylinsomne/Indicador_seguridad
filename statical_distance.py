from scipy.spatial import distance
from pca import pca
import pandas as pd
import numpy as np
import math
import string
"""
df={}
#distancia con respecto solo una variable: está mal?
fila_seleccionada1= df[df['codigofr'] == 123]
stdvar1= df['var1'].std()
x= fila_seleccionada1['var1']/stdvar1
fila_seleccionada2 = df[df['codigofr'] == 456]
stdvar2= df['var2'].std()
y=fila_seleccionada2['var2']/stdvar2
dist = distance.euclidean(x,y)
"""

def DSE(df, fpivote, val_fp1, val_fp2, lista_columnas_estudio):#distancia estadistica euclidiana : donde fpivote fila pivote,  valor fila pivote 1 y valor 2
    fila_seleccionada1 = df[df[fpivote] == val_fp1]
    fila_seleccionada2 = df[df[fpivote] == val_fp2]
    distancias = []
    for columna in lista_columnas_estudio:
            variable1 = fila_seleccionada1[columna]  
            variable2 = fila_seleccionada1[columna]  
            stdvar = df[columna].std()  # Calcular la desviación estándar de la columna 
            x = (variable1 / stdvar)  # se estandariza para manejar el respectivo peso
            y = (variable2 / stdvar)  # se estandariza para manejar el respectivo peso
            z = (x - y)**2  # Elevar al cuadrado la diferencia
            distancias.append(z)
    sumatoria = sum(distancias)
    dist = math.sqrt(sumatoria)
    return dist



def PCA(df,columnas_estudiadas,n_pruebas):#se tiene que encontrar la mayor varianza :)
        n_variables=len(columnas_estudiadas)
        n = df.shape[0]
        df_estudio=df[[columnas_estudiadas]].copy()
        df_stand=estandarizar_df(df_estudio)#primero se estandariza
        df_cov=df_stand.cov()
        identity_matrix = np.eye(df_cov.shape[0])
        lamb=encontrar_x(df_cov,identity_matrix) # lamb es el eigen_value mayor es decir, si se quiere estudiar en un futuro el otro sacado de la ecuación cuadrática se tiene que revisar esta función
        weights=eigenvector(df_cov,lamb)
        #sample_result=df['column_name'].mean()
        #sample_result = sum((x - sample_result)**2) / (n - 1)
        df_final=0
        return df_final


def encontrar_x(df_cov, identity_matrix):
    det = np.linalg.det(df_cov)
    
    # Calcular el determinante de la matriz - xI
    matriz=lambda x: df_cov - (identity_matrix *x) 
    det_matriz = lambda x: np.linalg.det(matriz)
    
    # Encontrar el valor de x tal que el determinante sea cero
    x = None
    if det != 0:
        # Calcular las raíces de la ecuación cuadrática
        raices = np.roots([det_matriz(1), -det])

        # Seleccionar la raíz con mayor valor absoluto
        x = max(raices, key=np.abs)
    return x

def eigenvector(def_cov, lamb):
        v_len=def_cov.shape[0] #se propone el tamaño del vector con respecto a la matriz de covarianzas es decir directamente del # de variables
        b=np.array([])
        for i in range(v_len):
              b=np.append(b,lamb)
        x = np.zeros(v_len)
        A_inv = np.linalg.inv(def_cov)
        x = A_inv @ b
        return x


def estandarizar_df(df):
      columnas= df.columns
      for i in columnas:
            media=df.i.mean()
            SD=df.i.sd()
            df[i]=(df[i] - media) / SD
      return df
# Ejemplo con tu matriz

if __name__=="__main__":
    matriz = np.array([[7, 23], [23, 3]])
    x = encontrar_x(matriz)

    print("El valor de x es:", x)


import requests

def get_coordinates(api_key, address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        lat = location['lat']
        lng = location['lng']
        return lat, lng
    else:
        print("No se pudo obtener las coordenadas.")
        return None, None

if __name__=="__main__":
    # Coloca tu clave de API de Google Maps aquí
    api_key = "TU_API_KEY"
    address = "Dirección en Bogotá, Colombia"
    lat, lng = get_coordinates(api_key, address)
    if lat is not None and lng is not None:
        print(f"Latitud: {lat}, Longitud: {lng}")
