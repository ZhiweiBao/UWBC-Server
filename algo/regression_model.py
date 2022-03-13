import numpy as np
from sklearn.linear_model import LinearRegression
import math

# Forecast number of new hubs
# We use number of calls as subjective increase or decrease
# 
# Feature: 0, Score: 27.33533
# Feature: 1, Score: 8.49028
# Feature: 2, Score: -26.05115
# Feature: 3, Score: 3.68872

class regression_model: 
    def __init__(self):

        # original data
        self.x = np.array([[9.69, 14.75, 15.89, 14.9], [0.95, 1.03, 1.34, 1.32],
             [0.23, 0.29, 0.32, 0.44], [0.25, 0.31, 0.39, 0.44],
             [1.04, 1.99, 1.42, 1.54], [0.75, 1.64, 1.87, 1.27],
             [0.27, 0.27, 0.33, 0.52], [0.56, 1.89, 1.24, 1.87],
             [1.27, 1.47, 1.55, 1.35], [0.47, 0.61, 0.69, 0.81]])
        self.y = np.array([40, 17, 3, 9, 15, 2, 14, 24, 34, 8])

        # modifeied data
        # self.x = np.array([[9.69, 14.75, 15.89, 14.9],
        #                    [0.95, 1.03, 1.34, 1.32],
        #                    [0.23, 0.29, 0.32, 0.44],
        #                    [0.25, 0.31, 0.39, 0.44],
        #                    [1.04, 1.99, 1.42, 1.54],
        #                    [0.47, 0.61, 0.69, 0.81]])
        # self.y = np.array([40, 17, 3, 9, 15, 8])

        # modified data 2
        # self.x = np.array([[9.69, 14.75, 14.9], [0.95, 1.03, 1.32],
        #      [0.23, 0.29, 0.44], [0.25, 0.31, 0.44],
        #      [1.04, 1.99, 1.54], [0.75, 1.64, 1.27],
        #      [0.27, 0.27, 0.52], [0.56, 1.89, 1.87],
        #      [1.27, 1.47, 1.35], [0.47, 0.61, 0.81]])
        # self.y = np.array([40, 17, 3, 9, 15, 2, 14, 24, 34, 8])
        
        self.model = LinearRegression()
        self.model.fit(self.x, self.y)
        importance = self.model.coef_

        # for i,v in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (i,v))

    def get_score(self):
        # returns R-squared coefficient as percentage
        return self.model.score(self.x, self.y) * 100

    def predict(self, x): 
        ####
        #   x data
        #   city name
        #   p1: number of people greater than sixty 
        #   p2: no certificate / household population
        #   p3: unemployed + not in labour force / household of labour force
        #   p4: lower than 60k of income
        #   normalize with average recorded population across features
        ####

        # Rules for smaller inputs
        # if x[2] < 5000: return 8
        # if x[2] > 5000 & x[2] < 15000: return 9
        # if x[2] > 15000 & x[2] < 25000: return 10
        # if x[2] > 25000 & x[2] < 50000: return 11

        self.normalize_inputs = {'vancouver': [100000, 100000, 100000, 100000], \
        'surrey': [120000, 120000, 120000, 120000], \
        'new westminster': [45000.0, 45000.0, 45000.0, 45000.0], \
        'richmond': [150000.0, 150000.0, 150000.0,150000.0], \
        'coquitlam': [35000, 35000, 35000.0, 35000],
        'delta': [25000, 25000, 25000, 25000],
        'victoria': [60000.0, 60000.0, 60000.0, 60000.0], \
        'abbotsford': [90000.0, 90000.0, 90000.0,90000.0]}

        normalized_x = []
        normalized_x.append(x[1] / self.normalize_inputs[x[0]][0])
        normalized_x.append(x[2] / self.normalize_inputs[x[0]][1])
        normalized_x.append(10000 / self.normalize_inputs[x[0]][2])
        normalized_x.append(10000 / self.normalize_inputs[x[0]][3])
        #x = 
        return math.floor(abs(self.model.predict([normalized_x])))