import warnings
from src.Circuit import Circuit
from src.Trained import Transformadores
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

# Instanciamos el transformador
transformadores = Transformadores()


def transform_dataset(df):
    """
    Transforma el dataset para poder ser usado por el modelo.
    args:
        df: dataframe con los datos a transformar.
    returns:
        dataframe con los datos transformados.
    """
    # Instanciamos el circuito
    circuit = Circuit(df)

    # Creamos el pipeline
    make_pipeline(
        circuit.validator.load_data(circuit.df),
        circuit.validator.validar_dataset(),
        circuit.validator.filtrar_localidades(),
        circuit.validator.binarize_cols(),
        circuit.validator.procesar_fecha(),
        circuit.transformer.load_data(circuit.validator.df),
        circuit.transformer.mapear_direcciones_viento(),
        circuit.transformer.mapear_localidades(),
        circuit.transformer.frequency_encode_categorical(),
        circuit.transformer.impute_knn(),
        circuit.transformer.estandarize(),
        circuit.transformer.obtain_pca(),
        circuit.transformer.obtain_new_features(),
    )

    # Retornamos el dataframe transformado
    return circuit.transformer.df


def select_model(model_to_use):
    """
    Selecciona el modelo a usar dependiendo de la opcion seleccionada.
    args:
        model_to_use: modelo a usar.
    returns:
        modelo a usar.
    """
    if model_to_use == 'clasif':
        return transformadores.best_classifier_model
    else:
        return transformadores.best_regressor_model

def make_predict(model, df, used_clasif=False):
    """
    Realiza la prediccion con el modelo seleccionado.
    args:
        model: modelo a usar.
        df: dataframe con los datos a predecir.
        used_clasif: booleano que indica si se usa clasificacion o regresion.
    returns:
        prediccion realizada.
    """
    prediccion = model.predict(df)
    if used_clasif:
        return 'Mañana Lloverá' if int(prediccion[0]) == 1 else 'No lloverá mañana'
    else:
        return f'Mañana lloverá {str(round(prediccion[0][0] if prediccion[0][0] > 0 else 0.0, 3))} mm'
