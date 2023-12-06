import pandas as pd
import numpy as np
import calendar
import Trained

class ValidateData:
    def __init__(self, df):
        self.df = self.validar_dataset(df)

    def validar_dataset(self, df):
        expected_columns = [
            'Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
            'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
            'Temp3pm', 'RainToday'
        ]
        expected_dtypes = {
            'Date': 'object', 'Location': 'object', 'MinTemp': 'float64',
            'MaxTemp': 'float64', 'Rainfall': 'float64', 'Evaporation': 'float64',
            'Sunshine': 'float64', 'WindGustDir': 'object', 'WindGustSpeed': 'float64',
            'WindDir9am': 'object', 'WindDir3pm': 'object', 'WindSpeed9am': 'float64',
            'WindSpeed3pm': 'float64', 'Humidity9am': 'float64', 'Humidity3pm': 'float64',
            'Pressure9am': 'float64', 'Pressure3pm': 'float64', 'Cloud9am': 'float64',
            'Cloud3pm': 'float64', 'Temp9am': 'float64', 'Temp3pm': 'float64',
            'RainToday': 'object'
        }

        # Validar columnas y tipos de datos
        if not all(col in df.columns for col in expected_columns):
            raise ValueError("El DataFrame no contiene todas las columnas esperadas.")
        
        if not all(df[col].dtype == expected_dtypes[col] for col in expected_columns):
            raise ValueError("Los tipos de datos de las columnas no son los esperados.")

        print('All columns are correct.')
        return df

    def filtrar_localidades(self, df):
        localidades = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
        df_c = df[df['Location'].isin(localidades)]
        df_c = df_c.reset_index(drop=True)
        return df_c

    def binarize_cols(self, df):
        df['RainToday'] = df['RainToday'].eq('Yes').mul(1)
        df[df['RainToday'].isna()] = np.nan
        return df

    def procesar_fecha(self, df):
        df_c = df.copy()
        df_c['Date'] = pd.to_datetime(df_c['Date'])
        df_c['Year'] = df_c['Date'].dt.year
        df_c['Month'] = df_c['Date'].dt.month
        df_c['Month'] = df_c['Month'].apply(lambda x: calendar.month_name[x])

        conditions = [
            (df_c['Month'].isin(['December', 'January', 'February'])),
            (df_c['Month'].isin(['March', 'April', 'May'])),
            (df_c['Month'].isin(['June', 'July', 'August'])),
            (df_c['Month'].isin(['September', 'October', 'November']))
        ]

        seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
        df_c['Season'] = np.select(conditions, seasons, default='Unknown')
        return df_c

    def obtener_dataframe_procesado(self):
        df_cf = self.filtrar_localidades(self.df)
        df_cb = self.binarize_cols(df_cf)
        df_cs = self.procesar_fecha(df_cb)
        return df_cs



df = pd.read_csv('weatherAUS.csv', usecols=range(1, 23))
df_pre = ValidateData(df).obtener_dataframe_procesado()

# class TransformData:
