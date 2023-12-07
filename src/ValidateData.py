import pandas as pd
import numpy as np
import calendar


class ValidateData:
    def __init__(self):
        pass

    def load_data(self, df):
        self.df = df

    def validar_dataset(self):
        expected_columns = [
            'Date', 'Location',
            'MinTemp', 'MaxTemp', 
            'Rainfall', 
            'Evaporation','Sunshine', 
            'WindGustDir', 'WindGustSpeed', 
            'WindDir9am', 'WindDir3pm',
            'WindSpeed9am', 'WindSpeed3pm', 
            'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 
            'Cloud9am', 'Cloud3pm', 
            'Temp9am', 'Temp3pm', 
            
        ]
        expected_dtypes = {
            'Date': 'object', 'Location': 'object',
            'MinTemp': 'float64', 'MaxTemp': 'float64',
            'Rainfall': 'float64',
            'Evaporation': 'float64', 'Sunshine': 'float64',
            'WindGustDir': 'object', 'WindGustSpeed': 'float64',
            'WindDir9am': 'object', 'WindDir3pm': 'object',
            'WindSpeed9am': 'float64', 'WindSpeed3pm': 'float64',
            'Humidity9am': 'float64', 'Humidity3pm': 'float64',
            'Pressure9am': 'float64', 'Pressure3pm': 'float64',
            'Cloud9am': 'float64', 'Cloud3pm': 'float64',
            'Temp9am': 'float64', 'Temp3pm': 'float64',
            
        }

        df_c = self.df[expected_columns]

        for col in expected_columns:
            if col in df_c.columns and df_c[col].dtype != expected_dtypes[col]:
                raise ValueError(f"El tipo de dato de la columna {col} no es el esperado.")

        print('Las columnas y tipos de datos mínimos son correctos.')
        return self.df

    def filtrar_localidades(self):
        localidades = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
        self.df = self.df[self.df['Location'].isin(localidades)]
        self.df = self.df.reset_index(drop=True)

    def binarize_cols(self):
        self.df['RainToday'] = self.df['RainToday'].eq('Yes').mul(1)
        self.df[self.df['RainToday'].isna()] = np.nan

    def procesar_fecha(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Month'] = self.df['Month'].apply(lambda x: calendar.month_name[x])

        conditions = [
            (self.df['Month'].isin(['December', 'January', 'February'])),
            (self.df['Month'].isin(['March', 'April', 'May'])),
            (self.df['Month'].isin(['June', 'July', 'August'])),
            (self.df['Month'].isin(['September', 'October', 'November']))
        ]

        seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
        self.df['Season'] = np.select(conditions, seasons, default='Unknown')
