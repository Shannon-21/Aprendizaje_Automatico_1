import pandas as pd
from src.Trained import Transformadores


class TransformData:
    """
    Clase que contiene las transformaciones de los datos.
    """
    def __init__(self):
        """
        Constructor de la clase.
        """
        self.models = Transformadores()

    def load_data(self, df):
        """
        Carga los datos a transformar.
        args:
            df: dataframe con los datos a transformar
        """
        self.df = df

    def frequency_encode_categorical(self):
        """
        Codifica las variables categoricas por frecuencia.
        """
        mapping_dict = self.models.encoder_cats

        for column, mapping in mapping_dict.items():
            self.df[column] = self.df[column].map(mapping)

    def mapear_direcciones_viento(self):
        """
        Mapea las direcciones del viento.
        """
        mapeo = {
            'N': 'N', 'NNE': 'N', 'NNW': 'N',
            'E': 'E', 'ENE': 'E', 'ESE': 'E',
            'S': 'S', 'SSE': 'S', 'SSW': 'S',
            'W': 'W', 'WSW': 'W', 'WNW': 'W',
            'NW': 'NW', 'NE': 'NE', 'SE': 'SE', 'SW': 'SW'
        }

        columnas_viento = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        for columna in columnas_viento:
            self.df[columna] = self.df[columna].replace(mapeo)

    def mapear_localidades(self):
        """
        Mapea las localidades.
        """
        mapeo = {'SydneyAirport': 'Sydney',
                'MelbourneAirport': 'Melbourne'}

        self.df['Location'] = self.df['Location'].replace(mapeo)
        self.df.reset_index(drop=True, inplace=True)

    def impute_knn(self):
        """
        Imputa los valores nulos con KNN.
        """
        knn_imputer_cats = self.models.knn_imputer_cats
        knn_imputer_nums = self.models.knn_imputer_nums
        nums = self.models.numerical_nulls
        cats = self.models.categorical_nulls

        self.df[nums] = knn_imputer_nums.transform(self.df[nums])
        self.df[cats] = knn_imputer_cats.transform(self.df[cats])

    def estandarize(self):
        """
        Estandariza los datos.
        """
        scaler = self.models.standar_scaler
        no_estandarizar = self.df[self.models.binaries]

        df_drop = self.df.drop(self.models.binaries + ['Year', 'Date'], axis=1)

        data_scaled = scaler.transform(df_drop)
        self.df = pd.DataFrame(data_scaled, columns=df_drop.columns)

        self.df[self.models.binaries] = no_estandarizar
    
    def obtain_pca(self):
        """
        Obtiene las componentes principales.
        """
        pca_model = self.models.pca_model

        reduced = pca_model.transform(self.df)
        df_reduced = pd.DataFrame(data=reduced, columns=[f"PC{i+1}" for i, _ in enumerate(self.df.columns)])
        df_reduced_first = df_reduced[[f"PC{i}" for i in range(1, 2)]]
        self.df = pd.concat([self.df, df_reduced_first], axis=1)

    def obtain_new_features(self):
        """
        Obtiene las nuevas caracteristicas.
        """
        self.df['Humidity_Index'] = self.df['Humidity9am'] * self.df['Humidity3pm']
        self.df['Evaporation_Index'] = self.df['Sunshine'] * self.df['Evaporation']
        self.df['Wind_Index'] = self.df['WindGustDir'] * self.df['WindGustSpeed']
        self.df['Light_Index'] = self.df['Sunshine'] * ((self.df['MaxTemp'] + self.df['MinTemp'])/2)
        self.df['Pressure_Index'] = self.df['Pressure3pm'] * self.df['Pressure9am'] 
        self.df['Rainfall_Index'] = self.df['Rainfall'] * self.df['Sunshine'] 
        self.df.drop(['Season', 'WindDir3pm', 'WindSpeed9am', 'Humidity9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday'], axis=1, inplace=True)