import numpy as np
import pandas as pd


def calc_time_diff (days_apart , lambda_decay = 0.2) :

    return np.exp(-lambda_decay * abs(days_apart))


def calc_env_weight (target_features , candidate_features , rf_weights_dict) :

    weather_distance = 0.0 

    for feature , importance in rf_weights_dict.items():

        diff =  abs(target_features[feature] - candidate_features[feature])
        weighted_diff = diff * importance
        weather_distance += weighted_diff


    env_weight = 1 / (weather_distance + 1e-5)

    return env_weight


class wKNN : 

    def __init__(self , k = 5 , lambda_decay = 0.2 , rain_threshold = 5.0):
        
        self.k = k
        self.lambda_decay = lambda_decay
        self.rain_threshold = rain_threshold


        self.data = None
        self.dist_matrix = None
        self.sim_matrix = None
        self.target_weights = None


    def fit(self, data , dist_matrix , sim_matrix , rf_weights_dict) :

        self.data = data.copy()
        self.dist_matrix = dist_matrix
        self.sim_matrix = sim_matrix
        self.rf_weights_dict = rf_weights_dict

        self.data['Date'] = pd.to_datetime(self.data['Date'])

        return self


    def predict(self , target_row , target_pm_col , known_pool): 

        target_station = target_row['Station']
        target_date = target_row['Date']
        target_rain = target_row['era5_precip_mm']

        
        

        




    