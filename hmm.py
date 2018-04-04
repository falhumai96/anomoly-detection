import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# HMM filters

# Filter by weekend
# Includes Friday, Saturday, Sunday
# e.g. weekends = filter_weekend("tests/train.txt", 30000)
def filter_weekend(csv_path, num_rows):
    # Get raw data (cannot be a copy)
    weekend = pd.read_csv(csv_path)

    if num_rows > 0:
        # Get only first n rows
        weekend = weekend[:num_rows]

    # Convert date column to python datetime object
    weekend['Date'] = pd.to_datetime(weekend['Date'])
    weekend['Hour'] = pd.to_datetime(weekend['Time'], format='%H:%M:%S').dt.hour
    weekend['Month'] = pd.to_datetime(weekend['Date']).dt.month
    

    # Create new column from Date: weekday number
    weekend['weekday'] = weekend['Date'].apply(lambda x: x.weekday())
    # Create new column from Date: weekday name
    weekend['weekday_name'] = weekend['Date'].apply(lambda x: x.weekday_name)
    
    # Filter by weekday >= 4
    weekend = weekend[weekend['weekday'] >= 4]
    return weekend

# Filter by weekday
# Includes Monday, Tuesday, Wednesday, Thursday, Friday
# e.g. weekday = filter_weekday("tests/train.txt", 30000)
def filter_weekday(csv_path, num_rows):
    # Get raw data (cannot be a copy)
    weekday = pd.read_csv(csv_path)

    if num_rows > 0:
        # Get only first n rows
        weekday = weekday[:num_rows]

    # Convert date column to python datetime object
    weekday['Date']  = pd.to_datetime(weekday['Date'])
    weekday['Hour']  = pd.to_datetime(weekday['Time'], format='%H:%M:%S').dt.hour
    weekday['Month'] = pd.to_datetime(weekday['Date']).dt.month

    # Create new column from Date: weekday number
    weekday['weekday'] = weekday['Date'].apply(lambda x: x.weekday())
    # Create new column from Date: weekday name
    weekday['weekday_name'] = weekday['Date'].apply(lambda x: x.weekday_name)
    
    # Monday = 0
    # Filter by weekday >= 4
    weekday = weekday[weekday['weekday'] <= 4]
    return weekday


# Filter by specific day
# weekday_name must have first letter capitalized e.g. "Friday" or "Wednesday"
# e.g. fridays = filter_specific_day("tests/train.txt", 30000, "Friday")
def filter_specific_day(csv_path, num_rows, weekday_name):
    # Get raw data (cannot be a copy)
    day = pd.read_csv(csv_path)
    
    if num_rows > 0:
        # Get only first n rows
        day = day[:num_rows]

    # Convert date column to python datetime object
    day['Date']  = pd.to_datetime(day['Date'])
    day['Hour']  = pd.to_datetime(day['Time'], format='%H:%M:%S').dt.hour
    day['Month'] = pd.to_datetime(day['Date']).dt.month

    # Create new column from Date: weekday number
    day['weekday'] = day['Date'].apply(lambda x: x.weekday())
    # Create new column from Date: weekday name
    day['weekday_name'] = day['Date'].apply(lambda x: x.weekday_name)

    # Filter by weekday_name
    day = day[day['weekday_name'] == weekday_name]
    return day

# Filter by month
def filter_months(dataframe, min_month, max_month):
    dataframe = dataframe[dataframe['Month'] >= min_month]
    dataframe = dataframe[dataframe['Month'] <= max_month]
    return dataframe

# Filter hours
# Removes all data that is greater than max_hour and less than min_hour
def filter_hours(dataframe, min_hour, max_hour):
    dataframe = dataframe[dataframe['Hour'] >= min_hour]
    dataframe = dataframe[dataframe['Hour'] <= max_hour]
    return dataframe

from pomegranate import *

# print("Splitting data into only Fridays")

# all_fridays = filter_specific_day("tests/train.txt", 0, "Friday")

# all_fridays = all_fridays.drop(columns=['Date', 'Time', 'Global_reactive_power', 'Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'weekday', 'weekday_name'])

# #all_fridays = all_fridays.values

print("Splitting data into only Weekdays")

weekdays = filter_weekday("tests/train.txt", 0)

print("Splitting Weekdays into only nights")

weekdays = filter_hours(weekdays, 0, 4)

print("Splitting into summer months only")

weekdays = filter_months(weekdays, 5, 7)

print("Dropping columns")

weekdays = weekdays.drop(columns=['Date', 'Time', 'Global_reactive_power', 'Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'weekday', 'weekday_name'])

print("Converting to nd_array")

weekdays = weekdays.values

print("Training HMM")
model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, #NormalDistribution,
                                       n_components=10,
                                       X=weekdays,
                                       algorithm="baum-welch",
                                       min_iterations=1,
                                       n_jobs=8)

# save to file
model_json = model.to_json()

import json
with open('model.json', 'w') as outfile:
    json.dump(model_json, outfile)

# save img

fig = model.plot()

fix.savefig("model.png")