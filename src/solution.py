import pickle
import numpy as np
from datetime import datetime

def circular_encode_timestamps(timestamps, format):
    
    # Convert timestamps to angles in radians
    angles = [2*np.pi*(datetime.strptime(str(ts), format).hour/24.0 + 
                       datetime.strptime(str(ts), format).minute/(24.0*60) + 
                       datetime.strptime(str(ts), format).second/(24.0*3600)) 
                       for ts in timestamps]
    
    # Convert angles to sine and cosine values
    sin_values = np.sin(angles)
    cos_values = np.cos(angles)

    # Compute arctangent of sin/cos ratio to get single-value encoding
    encodings = np.arctan2(sin_values, cos_values)
    
    # Return single-value encodings as a 1D array  
    return encodings



def predict_traffic_status(sensor, timestamp, flux, velocity):

    sensor_status_dict = {
        '10000028': ['Good', 'Medium Bad', 'Medium Good', 'Bad'],
        '10000029': ['Good', 'Medium Bad', 'Medium Good', 'Bad'],
        '10010172': ['Good', 'Bad', 'Medium Good', 'Medium Bad'],
        '10010173': ['Medium Bad', 'Bad', 'Good', 'Medium Good']
    }
    

    prediction_arguments = [[
        circular_encode_timestamps([timestamp], '%Y-%m-%d %H:%M:%S')[0],
        int(flux), 
        float(velocity)
    ]]

    print(prediction_arguments)

    file_path = './final_sensor_models/'+str(sensor)+'.pkl'

    with open(file_path, 'rb') as file:
        best_model = pickle.load(file)

        best_model_cluster_dictionary = ['Good', 'Medium Good', 'Medium Bad', 'Bad']

        cluster_index = best_model.predict(prediction_arguments)[0]

        return("Traffic status is: " + sensor_status_dict[sensor][cluster_index])
