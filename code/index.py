import pandas as pd
import warnings
from src.Circuit import Circuit
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')

reserva = pd.read_csv('data/X_reserva.csv', usecols=range(0, 22))
circuit = Circuit(reserva)

pipeline_process = make_pipeline(
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

ready_for_predict = circuit.transformer.df