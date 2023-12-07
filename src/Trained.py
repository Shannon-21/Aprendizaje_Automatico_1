import joblib

class Transformadores:

  def __init__(self):
    self.encoder_cats = {'Location': {'Sydney': 0.41639937494859774,
                          'Melbourne': 0.360144748745785,
                          'Canberra': 0.22345587630561725},
                        'WindGustDir': {'N': 0.14629932715039098,
                          'S': 0.11038370612838698,
                          'SSE': 0.08292416803055101,
                          'W': 0.07992362247681396,
                          'NW': 0.06937625022731406,
                          'SSW': 0.06719403527914167,
                          'WNW': 0.056373886161120206,
                          'NNW': 0.05328241498454264,
                          'WSW': 0.05073649754500818,
                          'ENE': 0.049099836333878884,
                          'NE': 0.046644844517184945,
                          'SW': 0.0463720676486634,
                          'NNE': 0.045553737043098744,
                          'E': 0.03855246408437898,
                          'SE': 0.03127841425713766,
                          'ESE': 0.026004728132387706},
                        'WindDir9am': {'N': 0.16150137741046833,
                          'W': 0.1431646005509642,
                          'WNW': 0.08496900826446281,
                          'NW': 0.08255853994490359,
                          'SSW': 0.06370523415977962,
                          'NNW': 0.06293044077134986,
                          'S': 0.060864325068870524,
                          'WSW': 0.059400826446280995,
                          'SW': 0.05268595041322314,
                          'SSE': 0.050878099173553716,
                          'SE': 0.043732782369146,
                          'NNE': 0.03934228650137741,
                          'ESE': 0.02703168044077135,
                          'E': 0.02591253443526171,
                          'NE': 0.023329889807162534,
                          'ENE': 0.017992424242424244},
                        'WindDir3pm': {'S': 0.12578616352201258,
                          'N': 0.09039832285115304,
                          'SSE': 0.08410901467505241,
                          'WNW': 0.07069182389937106,
                          'NW': 0.06658280922431865,
                          'E': 0.06607966457023061,
                          'NE': 0.0650733752620545,
                          'NNW': 0.06180293501048218,
                          'SSW': 0.05651991614255765,
                          'W': 0.05459119496855346,
                          'ENE': 0.054171907756813416,
                          'SE': 0.050985324947589096,
                          'WSW': 0.04410901467505241,
                          'ESE': 0.04318658280922432,
                          'NNE': 0.035136268343815516,
                          'SW': 0.03077568134171908},
                        'Month': {'May': 0.09392219754914055,
                          'March': 0.09046796611563451,
                          'January': 0.08701373468212846,
                          'June': 0.08635578583765112,
                          'August': 0.08586232420429311,
                          'October': 0.08495764454313677,
                          'July': 0.08216136195410807,
                          'September': 0.0810921950818324,
                          'November': 0.08076322065959371,
                          'December': 0.07813142528168435,
                          'April': 0.07730898922608767,
                          'February': 0.07196315486470926},
                        'Season': {'Autumn': 0.26169915289086276,
                          'Winter': 0.2543794719960523,
                          'Spring': 0.24681306028456287,
                          'Summer': 0.23710831482852207}}
    
    self.knn_imputer_nums = joblib.load('models\knn_imputer_nums_model.joblib')

    self.knn_imputer_cats = joblib.load('models\knn_imputer_cats_model.joblib')

    self.standar_scaler = joblib.load('models\standar_scaler_model.joblib')

    self.pca_model = joblib.load('models\pca_model.joblib')

    self.numerical_nulls = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
    self.categorical_nulls = ['WindGustDir','WindDir9am','WindDir3pm', 'RainToday', 'Cloud9am', 'Cloud3pm']

    self.binaries =  ['RainToday']
        