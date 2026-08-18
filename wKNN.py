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


    def fit(self, data , dist_matrix , sim_matrix , target_weights) :

        self.data = data.copy()
        self.dist_matrix = dist_matrix
        self.sim_matrix = sim_matrix
        self.target_weights = target_weights

        self.data['Date'] = pd.to_datetime(self.data['Date'])

        return self


    def predict(self , target_row , target_pm_col , known_pool): 

        target_station = target_row['Station']
        target_date = target_row['Date']
        target_rain = target_row['era5_precip_mm']

        rf_weights_dict  = self.target_weights[target_pm_col]


        date_min = target_date - pd.Timedelta(days = 14)
        date_max = target_date + pd.Timedelta(days = 14)

        candidates = known_pool[ (known_pool['Date'] >= date_min) & (known_pool['Date'] <= date_max)]

        if target_rain >= self.rain_threshold:

             filtered = candidates[candidates['era5_precip_mm'] >= self.rain_threshold]

        else :
            filtered = candidates[candidates['era5_precip_mm'] < self.rain_threshold]


        if len(filtered) >= self.k :
            candidates = filtered


        is_ring_2 = False

        if len(candidates) < self.k : 

            is_ring_2 = True
            target_month  = target_date.month
            candidates = known_pool[known_pool['Date'].dt.month == target_month]

            if target_rain >= self.rain_threshold:

                filtered = candidates[candidates['era5_precip_mm'] >= self.rain_threshold]
                
            else :
                filtered = candidates[candidates['era5_precip_mm'] < self.rain_threshold]


            if len(filtered) >= self.k :
                candidates = filtered



        weights = []
        pm_values = []


        for _ , candi_row in candidates.iterrows():

            candi_station = candi_row['Station']
            candi_date = candi_row['Date']

            geo_dist = self.dist_matrix.loc[target_station , candi_station]
            w_space = 1.0 / (geo_dist + 1e-5)
            w_sim = self.sim_matrix.loc[target_station , candi_station]


            if is_ring_2 :

                w_time = 1.0

            else: 
                days_apart = (target_date - candi_date).days
                w_time = calc_time_diff(days_apart , self.lambda_decay)


            w_env = calc_env_weight(target_row , candi_row , rf_weights_dict)


            w_total = w_space * w_sim * w_time * w_env

            weights.append(w_total)

            pm_values.append(candi_row[target_pm_col])



        weights = np.array(weights)
        pm_values = np.array(pm_values)
        top_k_indices = np.argsort(weights)[-self.k:]

        top_k_weights = weights[top_k_indices]
        top_k_pm = pm_values[top_k_indices]

        return np.sum(top_k_weights * top_k_pm) / np.sum(top_k_weights)



    def fill(self, target_columns) :

        print('Imputing missing values')
        completed_df = self.data.copy()

        for col in target_columns :

            missing_indices = completed_df[completed_df[col].isna()].index

            known_pool = completed_df[completed_df[col].notna()].copy()

            for idx in missing_indices:

                target_row = completed_df.loc[idx]
                predicted_val = self.predict(target_row , col , known_pool)
                completed_df.loc[idx,col] = predicted_val

                known_pool = completed_df[completed_df[col].notna()].copy()


            print(f'Imputation complete for column : {col}')


        print('Imputation complete')


        return completed_df


 
        




    