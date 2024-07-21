import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score, pairwise_distances, davies_bouldin_score
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
import copy
import csv

# Table columns

# 0 - MedidasCCVStatusId
# 1 - SensorCCVId
# 2 - VehicleTypeId
# 3 - Timestamp
# 4 - Fluxo
# 5 - Velocidade
# 6 - EstadodeTrafego


# Performance Metrics

# 1 - Elbow Method (not suited for computationally heavy algorithms)
# 2 - Silhouette Score = calculated using the mean intra-cluster distance and the mean nearest-cluster distance (the higher the better)
# 3 - Calinski-Harabaz Index = measure the distinctiveness between groups (the higher the better)
# 4 - Davies-Bouldin Index = average similarity of each cluster with its most similar cluster (the lower the better)



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

# Separates dataset's Timestamp column from '%Y-%m-%d %H:%M:%S' format, into two separate columns: 'Weekday' 
def normalize_timestamps(data):
    
    # Days of the week list
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Sort pandas dataframe by timestamps
    data = data.sort_values(by="Timestamp")
    
    # Group equal timestamps: sum the flow and average the velocity
    data = data.groupby('Timestamp').agg({
        'MedidasCCVStatusId': 'first',
        'SensorCCVId': 'first',
        'Fluxo': 'sum',
        'Velocidade': 'mean',
        'EstadodeTrafego': 'first',
        'OriginalTimestamp': 'first'
    }).reset_index()

    # Change timestamp strings to Datetime object type, facilitates following steps
    data['Timestamp'] =  [datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') for timestamp in data['Timestamp']];

    # Create column with the weekdays
    data['Weekday'] = [days.index(date_obj.strftime('%A')) for date_obj in data['Timestamp']];

    # Rewrite Timestamp column with just the hour
    aux = [date_obj.time() for date_obj in data['Timestamp']]
    data['Timestamp'] = aux

    return data

def getWeekdayResults(dayOfWeek, data):

    if(dayOfWeek < 0 or dayOfWeek > 6):
        return None
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    data = normalize_timestamps(data)
    
    data['Timestamp'] = circular_encode_timestamps(data['Timestamp'], '%H:%M:%S')
    weekday_data =  data[data['Weekday'] == dayOfWeek]

    filename = "../docs/model_performances/main/weekday_models/"+days[i]+"_results.csv"

    print("\n---------- MODEL RESULTS FOR "+days[i]+" ----------\n")
    train_and_test_models(weekday_data, filename)

def K_Means_flux_and_velocity(flux_and_velocity, file_writer):
    
    """ K-Means Algorithm - flux_and_velocity """

    # Execute the K-Means algorithm
    k = 4
    kmeans = KMeans(n_clusters=k, max_iter = 300, n_init = 10, random_state = 0).fit(flux_and_velocity)

    plt.scatter(flux_and_velocity['Fluxo'], flux_and_velocity['Velocidade'], c=kmeans.labels_, s = 10)
    plt.show()

    # The following section of code is meant to evaluate the clustering model evaluate the clustering model

    # Elbow Method
    cs = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(flux_and_velocity)
        cs.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), cs)
    plt.show()

    # Silhouette score
    silhouette_avg = silhouette_score(flux_and_velocity, kmeans.labels_)
    print("The average silhouette score for the flux_and_velocity dataset (K-Means) is :", silhouette_avg)

    # Calinski-Harabaz Index
    cal_index = metrics.calinski_harabasz_score(flux_and_velocity, kmeans.labels_)
    print("The Calinski-Harabaz Index for the flux_and_velocity dataset (K-Means) is :", cal_index)

    # Davies-Bouldin Index
    davies_index = davies_bouldin_score(flux_and_velocity, kmeans.labels_)
    print("The Davies-Bouldin Index for the flux_and_velocity dataset (K-Means) is :", davies_index)

    # Write a row to the csv file
    file_writer.writerow(["K-Means Algorithm - flux_and_velocity"])
    file_writer.writerows([["Silhouette score", "Calinski-Harabaz Index", "Davies-Bouldin Index"], [silhouette_avg, cal_index, davies_index]])

def K_Means_time_and_flux(time_and_flux, file_writer, originalTimestamps):

    # Execute the K-Means algorithm
    k = 4
    kmeans = KMeans(n_clusters=k, max_iter = 300, n_init = 10, random_state = 0).fit(time_and_flux)

    plt.scatter(originalTimestamps, time_and_flux['Fluxo'], c=kmeans.labels_, s = 10)
    plt.show()

    # The following section of code is meant to evaluate the clustering model evaluate the clustering model

    # Elbow Method
    cs = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(time_and_flux)
        cs.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), cs)
    plt.show()

    # Silhouette score
    silhouette_avg = silhouette_score(time_and_flux, kmeans.labels_)
    print("The average silhouette score for the time_and_flux dataset (K-Means) is :", silhouette_avg)

    # Calinski-Harabaz Index
    cal_index = metrics.calinski_harabasz_score(time_and_flux, kmeans.labels_)
    print("The Calinski-Harabaz Index for the time_and_flux dataset (K-Means) is :", cal_index)

    # Davies-Bouldin Index
    davies_index = davies_bouldin_score(time_and_flux, kmeans.labels_)
    print("The Davies-Bouldin Index for the time_and_flux dataset (K-Means) is :", davies_index)

    # Write rows to the csv file
    file_writer.writerow(["K-Means Algorithm - time_and_flux"])
    file_writer.writerows([["Silhouette score", "Calinski-Harabaz Index", "Davies-Bouldin Index"], [silhouette_avg, cal_index, davies_index]])

def K_Means_time_flux_and_velocity(time_flux_and_velocity, file_writer, originalTimestamps):
    
    # Execute the K-Means algorithm
    k = 4
    kmeans = KMeans(n_clusters=k, max_iter = 300, n_init = 10, random_state = 0).fit(time_flux_and_velocity)

    plt.scatter(originalTimestamps, time_flux_and_velocity['Velocidade'], c=kmeans.labels_, s = 10)
    plt.show()

    plt.scatter(originalTimestamps, time_flux_and_velocity['Fluxo'], c=kmeans.labels_, s = 10)
    plt.show()


    # The following section of code is meant to evaluate the clustering model evaluate the clustering model

    # Elbow Method
    cs = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(time_flux_and_velocity)
        cs.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), cs)
    plt.show()

    # Silhouette score
    silhouette_avg = silhouette_score(time_flux_and_velocity, kmeans.labels_)
    print("The average silhouette score for the time_flux_and_velocity (K-Means) dataset is :", silhouette_avg)

    # Calinski-Harabaz Index
    cal_index = metrics.calinski_harabasz_score(time_flux_and_velocity, kmeans.labels_)
    print("The Calinski-Harabaz Index for the time_flux_and_velocity (K-Means) dataset is :", cal_index)

    # Davies-Bouldin Index
    davies_index = davies_bouldin_score(time_flux_and_velocity, kmeans.labels_)
    print("The Davies-Bouldin Index for the time_flux_and_velocity (K-Means) dataset is :", davies_index)

    # Write rows to the csv file
    file_writer.writerow(["K-Means Algorithm - time_flux_and_velocity"])
    file_writer.writerows([["Silhouette score", "Calinski-Harabaz Index", "Davies-Bouldin Index"], [silhouette_avg, cal_index, davies_index]])

def time_K_Means_time_and_flux(time_and_flux, file_writer, originalTimestamps):
    
    time_series_kmeans = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=10, random_state=0).fit(time_and_flux)
    plt.scatter(originalTimestamps, time_and_flux['Fluxo'], c=time_series_kmeans.labels_, s = 10)
    plt.show()
    
    # The following section of code is meant to evaluate the clustering model evaluate the clustering model

    # Silhouette Score
    silhouette_avg = silhouette_score(time_and_flux, time_series_kmeans.labels_)
    print("The average silhouette score for the time_and_flux dataset (Timeseries K-Means) is :", silhouette_avg)

    # Calinski-Harabaz Index
    cal_index = metrics.calinski_harabasz_score(time_and_flux, time_series_kmeans.labels_)
    print("The Calinski-Harabaz Index for the time_and_flux dataset (Timeseries K-Means) is :", cal_index)

    # Davies-Bouldin Index
    davies_index = davies_bouldin_score(time_and_flux, time_series_kmeans.labels_)
    print("The Davies-Bouldin Index for the time_and_flux dataset (Timeseries K-Means) is :", davies_index)

    # Write rows to the csv file
    file_writer.writerow(["TimeSeries K-Means algorithm with Dynamic Time Warping - time_and_flux"])
    file_writer.writerows([["Silhouette score", "Calinski-Harabaz Index", "Davies-Bouldin Index"], [silhouette_avg, cal_index, davies_index]])

def time_K_Means_time_flux_and_velocity(time_flux_and_velocity, file_writer, originalTimestamps):
    
    time_series_kmeans = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=10, random_state=0).fit(time_flux_and_velocity)
    
    plt.scatter(originalTimestamps, time_flux_and_velocity['Fluxo'], c=time_series_kmeans.labels_, s = 10)
    plt.show()
    
    plt.scatter(time_flux_and_velocity['Velocidade'], time_flux_and_velocity['Fluxo'], c=time_series_kmeans.labels_, s = 10)
    plt.show()
    
    # The following section of code is meant to evaluate the clustering model evaluate the clustering model

    # Silhouette Score
    silhouette_avg = silhouette_score(time_flux_and_velocity, time_series_kmeans.labels_)
    print("The average silhouette score for the time_flux_and_velocity dataset (Timeseries K-Means) is :", silhouette_avg)

    # Calinski-Harabaz Index
    cal_index = metrics.calinski_harabasz_score(time_flux_and_velocity, time_series_kmeans.labels_)
    print("The Calinski-Harabaz Index for the time_flux_and_velocity dataset (Timeseries K-Means) is :", cal_index)

    # Davies-Bouldin Index
    davies_index = davies_bouldin_score(time_flux_and_velocity, time_series_kmeans.labels_)
    print("The Davies-Bouldin Index for the time_flux_and_velocity dataset (Timeseries K-Means) is :", davies_index)

    # Write rows to the csv file
    file_writer.writerow(["TimeSeries K-Means algorithm with Dynamic Time Warping - time_flux_and_velocity"])
    file_writer.writerows([["Silhouette score", "Calinski-Harabaz Index", "Davies-Bouldin Index"], [silhouette_avg, cal_index, davies_index]])

def gaussian_model_flux_and_velocity(flux_and_velocity, file_writer):
    gmm = GaussianMixture(n_components=4).fit(flux_and_velocity)
    color = gmm.predict(flux_and_velocity)

    plt.scatter(flux_and_velocity['Fluxo'], flux_and_velocity['Velocidade'], c=color, s = 10)
    plt.show()

    # The following section of code is meant to evaluate the clustering model evaluate the clustering model

    # Silhouette Score
    silhouette_avg = silhouette_score(flux_and_velocity, color)
    print("The average silhouette score for the flux_and_velocity dataset (Gaussian Mixture) is :", silhouette_avg)

    # Calinski-Harabaz Index
    cal_index = metrics.calinski_harabasz_score(flux_and_velocity, color)
    print("The Calinski-Harabaz Index for the flux_and_velocity dataset (Gaussian Mixture) is :", cal_index)

    # Davies-Bouldin Index
    davies_index = davies_bouldin_score(flux_and_velocity, color)
    print("The Davies-Bouldin Index for the flux_and_velocity dataset (Gaussian Mixture) is :", davies_index)
     
    # Write rows to the csv file
    file_writer.writerow(["Gaussian Mixture Model - flux_and_velocity"])
    file_writer.writerows([["Silhouette score", "Calinski-Harabaz Index", "Davies-Bouldin Index"], [silhouette_avg, cal_index, davies_index]])
    
def gaussian_model_time_and_flux(time_and_flux, file_writer, originalTimestamps):
    
    gmm = GaussianMixture(n_components=4).fit(time_and_flux)
    color = gmm.predict(time_and_flux)

    plt.scatter(originalTimestamps, time_and_flux['Fluxo'], c=color, s = 10)
    plt.show()

    # The following section of code is meant to evaluate the clustering model evaluate the clustering model

    # Silhouette Score
    silhouette_avg = silhouette_score(time_and_flux, color)
    print("The average silhouette score for the time_and_flux dataset (Gaussian Mixture) is :", silhouette_avg)

    # Calinski-Harabaz Index
    cal_index = metrics.calinski_harabasz_score(time_and_flux, color)
    print("The Calinski-Harabaz Index for the time_and_flux dataset (Gaussian Mixture) is :", cal_index)

    # Davies-Bouldin Index
    davies_index = davies_bouldin_score(time_and_flux, color)
    print("The Davies-Bouldin Index for the time_and_flux dataset (Gaussian Mixture) is :", davies_index)

    # Write rows to the csv file
    file_writer.writerow(["Gaussian Mixture Model - time_and_flux"])
    file_writer.writerows([["Silhouette score", "Calinski-Harabaz Index", "Davies-Bouldin Index"], [silhouette_avg, cal_index, davies_index]])

def gaussian_model_time_flux_and_velocity(time_flux_and_velocity, file_writer, originalTimestamps):
    
    gmm = GaussianMixture(n_components=4, init_params='k-means++').fit(time_flux_and_velocity)
    color = gmm.predict(time_flux_and_velocity)

    plt.scatter(originalTimestamps, time_flux_and_velocity['Fluxo'], c=color, s = 10)
    plt.show()
  
    plt.scatter(time_flux_and_velocity['Fluxo'], time_flux_and_velocity['Velocidade'], c=color, s = 10)
    plt.show()

    # The following section of code is meant to evaluate the clustering model evaluate the clustering model

    # Silhouette Score
    silhouette_avg = silhouette_score(time_flux_and_velocity, color)
    print("The average silhouette score for the time_flux_and_velocity dataset (Gaussian Mixture) is :", silhouette_avg)

    # Calinski-Harabaz Index
    cal_index = metrics.calinski_harabasz_score(time_flux_and_velocity, color)
    print("The Calinski-Harabaz Index for the time_flux_and_velocity dataset (Gaussian Mixture) is :", cal_index)

    # Davies-Bouldin Index
    davies_index = davies_bouldin_score(time_flux_and_velocity, color)
    print("The Davies-Bouldin Index for the time_flux_and_velocity dataset (Gaussian Mixture) is :", davies_index)

    # Write rows to the csv file
    file_writer.writerow(["Gaussian Mixture Model - time_flux_and_velocity"])
    file_writer.writerows([["Silhouette score", "Calinski-Harabaz Index", "Davies-Bouldin Index"], [silhouette_avg, cal_index, davies_index]])

def train_and_test_models(data, file_path):

    file = open(file_path, 'a')

    # create the csv writer
    file_writer = csv.writer(file)

    originalTimestamps = data['OriginalTimestamp']

    """ Create the subsets of data """

    # Reduce dataset to 'Fluxo' and 'Velocidade' columns
    flux_and_velocity = data[['Fluxo', 'Velocidade']]
    # Reduce dataset to 'Fluxo' and 'Timestamp' columns
    time_and_flux = data[['Timestamp', 'Fluxo']]
    # Reduce dataset to 'Fluxo', 'Timestamp' and 'Velocidade' columns
    time_flux_and_velocity = data[['Timestamp', 'Fluxo', 'Velocidade']]


    K_Means_flux_and_velocity(flux_and_velocity, file_writer)
    K_Means_time_and_flux(time_and_flux, file_writer, originalTimestamps)
    K_Means_time_flux_and_velocity(time_flux_and_velocity, file_writer, originalTimestamps)

    time_K_Means_time_and_flux(time_and_flux, file_writer, originalTimestamps)
    time_K_Means_time_flux_and_velocity(time_flux_and_velocity, file_writer, originalTimestamps)

    gaussian_model_flux_and_velocity(flux_and_velocity, file_writer)
    gaussian_model_time_and_flux(time_and_flux, file_writer, originalTimestamps)
    gaussian_model_time_flux_and_velocity(time_flux_and_velocity, file_writer, originalTimestamps)

    # close the file
    file.close()



if __name__ == "__main__":
    
    data = pd.read_csv("../dataset/data_with_velocity.csv")

    data = data.sort_values(by="Timestamp")
    data['OriginalTimestamp'] = data['Timestamp'].copy()
    grouped_data = data.groupby('SensorCCVId')

    for value in data['SensorCCVId'].unique():
        sensor_data = grouped_data.get_group(value)
        sensor_data.to_csv('../sensor_datasets/'+str(value)+'.csv', index=False, na_rep='NULL')





    """ Separate model training into days of the week """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for i in range(0,7):
        getWeekdayResults(i, data)


    """ Model training using only encoded timestamps (no weekday distinction) """

    print("\n---------- ALL MODEL RESULTS ----------\n")
    data['Timestamp'] = circular_encode_timestamps(data['Timestamp'], '%Y-%m-%d %H:%M:%S')
    train_and_test_models(data, "../docs/model_performances/main/all_results.csv")
