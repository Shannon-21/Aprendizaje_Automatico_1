import warnings
from src.Circuit import Circuit
from src.Trained import Transformadores
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

transformadores = Transformadores()


def transform_dataset(df):
    circuit = Circuit(df)

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

    return circuit.transformer.df


def select_model(model_to_use):
    if model_to_use == 'clasif':
        return transformadores.best_classifier_model
    else:
        return transformadores.best_regressor_model

def make_predict(model, df, used_clasif=False):
    prediccion = model.predict(df)
    if used_clasif:
        return 'Mañana Lloverá' if int(prediccion[0]) == 1 else 'No lloverá mañana'
    else:
        return f'Mañana lloverá {str(round(prediccion[0][0], 3))} mm'
