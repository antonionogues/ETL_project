# Practica final de Extracción, Transformación y Carga de datos

# Antonio Nogués Podadera

# **La práctica se ha realizado mediante SQL.**

# Librerias

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect
import re 
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Conexión con la base de datos

# Objeto que representa la conexión a la base de datos

engine = create_engine('sqlite:///airbnb.sqlite') 

# Consulta para saber qué tablas tiene la base de datos

inspector = inspect(engine)
print(inspector.get_table_names())

# EXTRACCION

## Tabla Listings

query_1 = """
SELECT Listings.id, Hoods.neighbourhood_group, Listings.price, Listings.number_of_reviews, Listings.review_scores_rating, Listings.room_type
FROM Listings
JOIN Hoods 
ON Listings.neighbourhood_cleansed = Hoods.neighbourhood
"""

table_listings = pd.read_sql(query_1, engine)  
table_listings.head()


## Tabla Reviews

query_2 = """
SELECT COUNT(Reviews.id) AS number_reviews, Hoods.neighbourhood_group, strftime('%Y-%m', Reviews.date) AS month
FROM Reviews
INNER JOIN Listings
ON Listings.id=Reviews.listing_id
INNER JOIN Hoods
ON Listings.neighbourhood_cleansed=Hoods.neighbourhood
WHERE strftime('%Y', Reviews.date) NOT LIKE 2010
GROUP BY Hoods.neighbourhood_group, month
"""

table_reviews = pd.read_sql(query_2, engine)  
table_reviews.head()

# TRANSFORMACION

## Transformación tabla `Listings` 

### 3. Transformación de la columna `price`

# Queremos eliminar el simbolo $ y la coma de los elementos de la columna "price"

# Selección de la columna en la que quiero aplicar la expresión regular (price)

col_price = table_listings['price']

# Aplicamos la expresión regular para eliminar el símbolo $

col_price = col_price.apply(lambda x: re.sub(r'\$', '', x))

# Aplicamos la expresión regular para eliminar la coma

col_price = col_price.apply(lambda x: re.sub(r',', '', x))

# Guardamos los cambios en el dataframe

table_listings['price'] = col_price

table_listings.head()


# Ahora podemos convertir la columna `price` a tipo numérico (float)

table_listings['price']= pd.to_numeric(table_listings['price'])

# Comprobación de que se ha convertido correctamente

table_listings['price'].dtype

### 4. Imputación de valores missing de `number_of_reviews` y `review_scores_rating`

# Imputación de valores missing sobre `number_of_reviews`

# Agrupamos los datos por la columna 'room_type'
groups = table_listings.groupby('room_type')

# Semilla para que los resultados sean reproducibles
random.seed(12345)

# Iteración sobre cada grupo
for name, group in groups:
  # Seleccionamos los valores de la columna 'number_of_reviews' que no son NA
  values = group['number_of_reviews'].dropna().values

  # Iteración sobre cada fila del grupo
  for index, row in group.iterrows():
    # Si el valor de la columna 'number_of_reviews' es NA, rellenamos con un valor aleatorio
    if pd.isnull(row['number_of_reviews']):
      table_listings.at[index, 'number_of_reviews'] = random.choice(values)

# Imputación de valores missing sobre `review_scores_rating`

# Semilla para que los resultados sean reproducibles
random.seed(12345)

# Iteración sobre cada grupo
for name, group in groups:
  # Seleccionamos los valores de la columna 'review_scores_rating' que no son NA
  values = group['review_scores_rating'].dropna().values

  # Iteración sobre cada fila del grupo
  for index, row in group.iterrows():
    # Si el valor de la columna 'review_scores_rating' es NA, rellenamos con un valor aleatorio
    if pd.isnull(row['review_scores_rating']):
      table_listings.at[index, 'review_scores_rating'] = random.choice(values)

# Se muestran los resultados de las dos columnas transformadas

table_listings.head()

### 5. Agregación de datos

# Agrupamos el dataframe por distrito y tipo de alojamiento

group_df = table_listings.groupby(['neighbourhood_group', 'room_type'])

# Esta función calcula la media ponderada (review_scores_rating ponderado con number_of_reviews)

def calculo_media_ponderada(group):
  media_ponderada = (group["review_scores_rating"] * group["number_of_reviews"]).sum() / group["number_of_reviews"].sum()
  return media_ponderada


# Cálculo de la media ponderada

mean = group_df.apply(calculo_media_ponderada)
mean = pd.DataFrame(mean).reset_index().rename(columns={0: 'nota_media'})


# Cálculo del precio mediano

median_price = group_df['price'].median()
median_price = pd.DataFrame(median_price).reset_index().rename(columns={'price': 'precio_mediano'})

# Cálculo del número de alojamientos

count_id = group_df['id'].count()
count_id = pd.DataFrame(count_id).reset_index()

# Unión de dataframes

new_listings = pd.merge(count_id, median_price, on=['neighbourhood_group', 'room_type'])
new_listings = pd.merge(median_price, mean, on=['neighbourhood_group', 'room_type'])

new_listings.head()

## Transformación tabla `Reviews`

### 6. Predicciones 

# Últimos valores de cada distrito

last_value = table_reviews.groupby('neighbourhood_group').tail(1)

# Se incrementa el mes de julio a agosto

last_value['month'] = last_value['month'].str.replace(r'2021-07', '2021-08')
last_value

# Unión de dataframes

new_reviews = pd.concat([table_reviews, last_value]).sort_values(by=['neighbourhood_group', 'month'])
new_reviews.head(5)

### 7. Casos en los que no hay datos

# Creación de todas las fechas posibles entre 2011-01 y 2021-08

month = pd.date_range('2011-01', '2021-08', freq='MS')

# Posibles distritos

neighbourhood_group = pd.Series(last_value['neighbourhood_group'].unique())

# Unión de los dataframes de fechas y distritos creados anteriormente. 

combinations = pd.MultiIndex.from_product([month, neighbourhood_group], names=['month', 'neighbourhood_group']).to_frame().reset_index(drop=True)
combinations.head()

# Modificación del formato de la columna 'month' a año y mes 

combinations['month'] = combinations['month'].apply(lambda x: datetime.strftime(x, '%Y-%m'))
combinations.head()

# Unión de dataframes
new_reviews = pd.merge(combinations, new_reviews, on=['month', 'neighbourhood_group'], how='outer')

# Se ordenan los valores por distrito y mes
new_reviews = new_reviews.sort_values(by=['neighbourhood_group', 'month'])

new_reviews.head()

# Tenemos valores nulos en la columna number_reviews puesto que en esa fecha nueva 
# que se ha creado no hay ningún review. Para solucionarlo se imputara un 0 a estos valores. 


new_reviews['number_reviews'] = new_reviews['number_reviews'].fillna(0)
new_reviews.head()

### 8. CARGA

## Carga del dataframe `new_listings` a la base de datos

# Carga del dataframe en la base de datos. 

# El argumento "if_exists='replace'" se ha añadido para que al volver a ejecutar el código no de error 
# al intentar crear una tabla que ya existe.

new_listings.to_sql('new_listings', engine, if_exists='replace', index=True)

# Comprobación de la carga: **Consulta de los datos de la tabla "new_listings"**

check_listings_query = """
SELECT * 
FROM new_listings LIMIT 10
"""

table_reviews = pd.read_sql(check_listings_query, engine)  
table_reviews.head()

## Carga del dataframe `new_reviews` a la base de datos

# Carga del dataframe en la base de datos

# El argumento "if_exists='replace'" se ha añadido para que al volver a ejecutar el código 
# no de error al intentar crear una tabla que ya existe.

new_reviews.to_sql('new_reviews', engine, if_exists='replace', index=True)

# Comprobación de la carga: **Consulta de los datos de la tabla "new_reviews"**

check_reviews_query = """
SELECT * 
FROM new_reviews LIMIT 10
"""

table_reviews = pd.read_sql(check_reviews_query, engine)  
table_reviews.head()






