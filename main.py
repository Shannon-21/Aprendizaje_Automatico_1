import streamlit as st
import pandas as pd
from code.index import transform_dataset, select_model, make_predict


st.title('Lluvia en australia')
st.subheader('Este trabajo consiste en predecir si lloverá mañana en Australia.')

st.write('Formulario para ingresar la data:')

df = pd.read_csv('data/df_filtrado.csv', usecols=lambda x: x not in ['Unnamed: 0', 'RainTomorrow', 'RainfallTomorrow'])
columnas = df.columns

cantidad_columnas_streamlit = 2
columnas_streamlit = st.columns(cantidad_columnas_streamlit)

inputs = []

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


modelo = st.selectbox('¿Que quieres predecir?', ['¿Cuanto lloverá mañana?', '¿Lloverá mañana?'])

if st.button('Predecir', key='predict', help='Predice si lloverá el día de mañana'):
    new_df = pd.DataFrame(columns=columnas)

    for columna, input_value in zip(columnas, inputs):
        new_df[columna] = [input_value]

    transformed_df = transform_dataset(new_df)
    model = select_model('clasif' if modelo == '¿Lloverá mañana?' else 'regresion')
    prediction = make_predict(model, transformed_df, used_clasif=modelo == '¿Lloverá mañana?')
    st.write(f'Predicción del modelo: {prediction}')
