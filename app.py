from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():  # put application's code here
    # {'City': "Vancouver", 'Locations': [[4,4], [2,3], [1,2]]}

    result = np.array([[-123.101787, 49.226260],
                       [-123.134670, 49.208772],
                       [-123.058691, 49.214179],
                       [-123.065507, 49.225525],
                       [-123.067150, 49.245800],
                       [-123.049046, 49.260693]])
    r = result.tolist()
    send = []
    for item in r:
        send.append(item)
    return jsonify(send)


if __name__ == '__main__':
    app.run()
