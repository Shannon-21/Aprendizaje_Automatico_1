from src.TransformData import TransformData
from src.ValidateData import ValidateData

class Circuit:
    """
    Clase que contiene el circuito de transformacion de datos.
    """
    def __init__(self, df):
        """
        Constructor de la clase.
        args:
            df: dataframe con los datos a transformar.
        """
        self.df = df
        self.validator = ValidateData()
        self.transformer = TransformData()