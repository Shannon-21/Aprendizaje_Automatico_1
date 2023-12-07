from src.TransformData import TransformData
from src.ValidateData import ValidateData

class Circuit:
    def __init__(self, df):
        self.df = df
        self.validator = ValidateData()
        self.transformer = TransformData()