from __future__ import division

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework_csv.parsers import CSVParser
from rest_framework.settings import api_settings
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
from rest_framework.parsers import FileUploadParser
import json
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams['figure.max_open_warning'] = 40
from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
style.use('fivethirtyeight')
# %matplotlib inline

def moving_average(data, window_size):
    """ Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
    -----
            data (pandas.Series): independent variable
            window_size (int): rolling window size

    Returns:
    --------
            ndarray of linear convolution
    """

    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def explain_anomalies(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using stationary standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies

    """
    avg = moving_average(y, window_size).tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    index_val = list(y.index)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i in zip(index_val, y, avg)
              if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}


def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using rolling standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan, testing_std_as_df.ix[window_size - 1]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    index_val = list(y.index)
    return {'stationary standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i, rs_i in zip(index_val,
            y, avg_list, rolling_std) if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}


# This function is repsonsible for displaying how the function performs on the given dataset.
def plot_results(x, y, window_size, sigma_value=1, text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
    """ Helps in generating the plot and flagging the anamolies. Supports both moving and stationary standard deviation. Use the 'applying_rolling_std' to switch
        between the two.
    Args:
    -----
        x (pandas.Series): dependent variable
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma_value (int): value for standard deviation
        text_xlabel (str): label for annotating the X Axis
        text_ylabel (str): label for annotatin the Y Axis
        applying_rolling_std (boolean): True/False for using rolling vs stationary standard deviation
    """
    plt.figure(figsize=(15, 8))
    plt.plot(x, y, "k.")
    y_av = moving_average(y, window_size)
    plt.plot(x, y_av, color='green')
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)

    # Query for the anomalies and plot the same
    events = {}
    if applying_rolling_std:
        events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
    else:
        events = explain_anomalies(y, window_size=window_size, sigma=sigma_value)

    x_anomaly = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
    y_anomaly = np.fromiter(events['anomalies_dict'].values(), dtype=float, count=len(events['anomalies_dict']))

    plt.plot(x_anomaly, y_anomaly, "r*", markersize=12)

    # add grid and lines and enable the plot
    plt.grid(True)
    plt.show()


# def find_anomaly(data=data, sub_division='ANDAMAN & NICOBAR ISLANDS', allowed_deviation=25):
#     sub_data = data[data['SUBDIVISION'] == sub_division]
#     train = sub_data[sub_data['YEAR'] < 2000]
#     test = sub_data[sub_data['YEAR'] > 1999]



class FileView(APIView):
    # parser_classes = (CSVParser,) + tuple(api_settings.DEFAULT_PARSER_CLASSES)
    serializer_class = FileSerializer
    # parser_classes = (FileUploadParser,)

    def post(self, request, *args, **kwargs):

        file_name = request.data["csv_file"]
        dic = {}
        dic["file_name"]= "ok"

        data = pd.read_csv(file_name)
        data.head()
        # print(data['YEAR'].min(), data['YEAR'].max())
        x = data['SUBDIVISION'].unique()
        # print(x, len(x))
        col = [i for i in data.columns if i not in ['SUBDIVISION', 'YEAR', 'ANNUAL']]
        data = data.drop(col, axis=1)
        data.info()
        data['ANNUAL'] = data['ANNUAL'].fillna(data['ANNUAL'].median())
        data.info()
        data.groupby(['SUBDIVISION']).plot(x='YEAR', y='ANNUAL')
        # checking one plot manually, for 'ANDAMAN & NICOBAR ISLANDS'
        sample = data[data['SUBDIVISION'] == 'ANDAMAN & NICOBAR ISLANDS']
        plt.plot(sample.YEAR, sample.ANNUAL)
        # print(sample.info())
        sample.head()
        sample.index = sample.YEAR
        sample.head()
        x = sample['YEAR']
        y = sample['ANNUAL']
        y.head()
        # plot the results
        plot_results(x, y=y, window_size=12, text_xlabel="Year", sigma_value=2, text_ylabel="Rainfall")
        events = explain_anomalies(y, window_size=12, sigma=2)

        # Display the anomaly dict
        # print("Information about the anomalies model:{}".format(events))
        # dividing the data into train and test sets
        train = sample[sample['YEAR'] < 2000]  # using first 100 years data for training
        # print(train.head())
        test = sample[sample['YEAR'] > 1999]  # using next 15 yeara data for testing
        # print(test.head())

        # Note: Ignoring the missing years in train and test set, as our model is independent of continuity of data interval
        train.index = pd.DatetimeIndex(train['YEAR'].map(lambda x: pd.Timestamp(str(x))))
        # print(train.head())
        test.index = pd.DatetimeIndex(test['YEAR'].map(lambda x: pd.Timestamp(str(x))))
        # print(test.head())
        print(events)

        return Response(events, status=status.HTTP_201_CREATED)

