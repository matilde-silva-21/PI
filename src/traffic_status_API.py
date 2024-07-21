from flask import Flask
from flask import request
import solution
from waitress import serve

app = Flask(__name__)

@app.route("/", methods=["POST"])
def test():
    timestamp = request.form.get('Timestamp')
    fluxo = request.form.get('Fluxo')
    velocidade = request.form.get('Velocidade')
    sensor = request.form.get('SensorID')

    traffic_status = solution.predict_traffic_status(sensor, timestamp, fluxo, velocidade)

    return (traffic_status, 200)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)