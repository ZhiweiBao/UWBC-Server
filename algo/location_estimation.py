import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class LocationEstimator:
    def __init__(self):
        self.__data_dir = 'D:\JetBrainsProjects\PycharmProjects\\flaskProject\data\cordinates_partner_transport.csv'
        self.__load_data()

    def __load_data(self):
        self.__data = pd.read_csv(self.__data_dir)

    def location_estimation(self, city, num_location):
        print(city)
        if not city or city not in self.__data.columns or num_location < 1:
            return np.array([])
        cordinates = self.__data[city].dropna()
        cord = np.array([[0.0, 0.0] for _ in range(len(cordinates))])
        for i in range(len(cordinates)):
            cord[i][0], cord[i][1] = cordinates[i].split(',')

        X_train = cord
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_train)

        X = X_scaled
        kmeans = KMeans(n_clusters=num_location, random_state=0).fit(X)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        return centers
