from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
from algo import location_estimation
from algo.regression_model import regression_model

app = Flask(__name__)
CORS(app)


@app.route('/feature_forecast/<string:city_name>/<string:pop_over60>/<string:unemployed>')
def forecast_func(city_name, pop_over60, unemployed):
    model = regression_model()
    # x_input is of the structure: ['Vancouver', population over 60, unemployed]
    x_input = [city_name, int(pop_over60), int(unemployed), 0, 0]
    result = model.predict(x_input)
    # result is a number showing the number of food hubs
    estimator = location_estimation.LocationEstimator()
    locations = estimator.location_estimation(city_name, result).tolist()
    if len(locations) == 0:
        return jsonify(None)
    res = [item for item in locations]
    list_of_lat_and_long = res
    return jsonify(list_of_lat_and_long)


@app.route('/hub_number/<string:city_name>/<string:location_num>')
def hub_number(city_name, location_num):
    estimator = location_estimation.LocationEstimator()
    locations = estimator.location_estimation(city_name, int(location_num)).tolist()
    if len(locations) == 0:
        return jsonify(None)
    res = [item for item in locations]
    return jsonify(res)


if __name__ == '__main__':
    app.run()
