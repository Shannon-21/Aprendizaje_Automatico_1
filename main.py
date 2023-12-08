import streamlit as st
import pandas as pd
from code.index import transform_dataset, select_model, make_predict

# Configuramos el titulo de la pagina y el icono
st.set_page_config(page_title='Lluvia en Australia', page_icon=':⛈️:')

# Configuramos el titulo y subtitulo de la pagina
st.markdown("<h1 style='text-align: center;'>Lluvia en Australia</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Este trabajo consiste en predecir si lloverá mañana en Australia.</h4>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center;'>Formulario para ingresar la data</h5>", unsafe_allow_html=True)


# Cargamos el dataset
df = pd.read_csv('data/df_filtrado.csv', usecols=lambda x: x not in ['Unnamed: 0', 'RainTomorrow', 'RainfallTomorrow'])

# Obtenemos las columnas del dataset
columnas = df.columns

# Creamos los inputs para el formulario
cantidad_columnas_streamlit = 2
columnas_streamlit = st.columns(cantidad_columnas_streamlit)

inputs = []
# Creamos los inputs para el formulario
for i in range(len(columnas)):
    columna = columnas[i]
    if columna.startswith('Cloud'):
        df[columna] = df[columna]
        inputs.append(columnas_streamlit[i % cantidad_columnas_streamlit].selectbox(columna, sorted(df[df[columna].notna()][columna].unique())))
        continue
    if columna == 'Date':
        df[columna] = pd.to_datetime(df[columna])
        inputs.append(columnas_streamlit[i % cantidad_columnas_streamlit].date_input(columna, min_value=df[columna].min(), max_value=df[columna].max(), value=df[columna].mean()))
        continue
    if df[columna].dtype == 'object':
        inputs.append(columnas_streamlit[i % cantidad_columnas_streamlit].selectbox(columna, df[df[columna].notna()][columna].unique()))
    else:
        inputs.append(columnas_streamlit[i % cantidad_columnas_streamlit].number_input(columna, min_value=float(df[columna].min()), max_value=float(df[columna].max()), value=float(df[columna].mean())))

# Creamos el selectbox para seleccionar el modelo
modelo = st.selectbox('¿Que quieres predecir?', ['¿Cuanto lloverá mañana?', '¿Lloverá mañana?'])

# Creamos el boton para predecir
if st.button('Predecir', key='predict', help='Predice si lloverá el día de mañana', use_container_width=True):
    # Creamos el dataframe con los inputs
    new_df = pd.DataFrame(columns=columnas)

    for columna, input_value in zip(columnas, inputs):
        new_df[columna] = [input_value]

    # Transformamos el dataset
    transformed_df = transform_dataset(new_df)

    # Seleccionamos el modelo
    model = select_model('clasif' if modelo == '¿Lloverá mañana?' else 'regresion')

    # Realizamos la prediccion
    prediction = make_predict(model, transformed_df, used_clasif=modelo == '¿Lloverá mañana?')

    # Mostramos la prediccion
    st.markdown(f"<h3 style='text-align: center;'>Según el modelo: {prediction}</h3>", unsafe_allow_html=True)


# Mostramos los autores
st.markdown("<h3 style='text-align: center;'>Autores</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Constantino Ferrucci y Fabio Giampaoli</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>2023</p>", unsafe_allow_html=True)