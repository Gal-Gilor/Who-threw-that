import os
import joblib as jb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

from typing import Optional, List, Dict, Union

from scipy.stats import randint, shapiro
from scipy.integrate import cumtrapz

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor

import xgboost as xgb


# graph styling
plt.style.use('seaborn-dark')
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16


# EDA Functions


def shapiro_wilk(data: Union[List, np.array, pd.Series], alpha: Optional[float] = 0.05) -> None:
    '''
    Checks whether a sample of data distributes normally according to the Shapiro Wilks normality test
    input:
        data: list of a sample of observations
        alpha: float, the significance level, 0.05 by default.
    '''
    statistic, p = shapiro(data)

    print(f'Statistics: {statistic}\nP_value: {p}\nalpha: {alpha}')
    if p > alpha:
        print('Sample looks Gaussian. Failed to reject the null hypothesis')
    else:
        print('Sample does not look Gaussian reject the null hypothesis)')
    return


def create_hist(data: Union[pd.Series, List], title: Optional[str] = None, xlabel: Optional[str] = None):
    '''
    plots histograms
    inputs:
        data: an array of to plot 
        title: optional string, graph title
        xlabel: optional string, x label title
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data)

    # add the labels & remove borders
    ax.set_title(title)
    ax.set_ylabel('Frequency')
    ax.set_xlabel(xlabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def create_line(x: Union[pd.Series, List], y: Union[pd.Series, List], title: Optional[str] = None,
                xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    '''
    plots histograms
    inputs:
        x: an array of to plot
        y: an array of to plot
        title: optional string, graph title
        xlabel: optional string, x label title
        ylabel: optional string, y label title
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y)

    # add the labels & remove borders
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def create_scatter(x: Union[np.array, pd.Series, List], y: Union[np.array, pd.Series, List],
                   title: Optional[str] = None, xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None) -> None:
    '''
    plots histograms
    inputs:
        x: an array of to plot
        y: an array of to plot
        title: optional string, graph title
        xlabel: optional string, x label title
        ylabel: optional string, y label title
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)

    # add the labels & remove borders
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return


def three_dimonsional_scatter(x: Union[np.array, List, pd.Series],
                              y: Union[np.array, List, pd.Series],
                              z: Union[np.array, List, pd.Series],
                              colorscale: Optional[str] = 'Viridis') -> None:
    '''
    Create a 3 dimensional scatterplpt
    input:
        x: list, data to plot against the x axis
        y: list, data to plot against the y axis
        z: list, data to plot against the z axis
        colorscale: string, one of the plotly's available colormaps, Viridis by default
    '''
    # configure the trace.
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker={
            'size': 3,
            'opacity': 0.7,
            'color': x,
            'colorscale': colorscale,
            'colorbar': dict(thickness=10)
        }
    )
    data = [trace]

    # configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(plot_figure)
    return


# Dataset Processing Functions


def rename_observation_columns(df: pd.DataFrame, columns: Optional[Dict] = {
    "Time_s_": "time",
    "Acc_x_m_s_2_": "acc_x",
    "Acc_y_m_s_2_": "acc_y",
    "Acc_z_m_s_2_": "acc_z",
    "Gyro_x_1_s_": "gyro_x",
    "Gyro_y_1_s_": "gyro_y",
    "Gyro_z_1_s_": "gyro_z",
}) -> pd.DataFrame:
    '''
    Rename the column names in the individual throws dataframes
    Note: Specific to this kaggle dataset columns
    inputs:
        df: dataframe, raw imu file data
    '''
    return df.rename(columns=columns)


def filter_columns(df: pd.DataFrame, regex: Optional[str] = 'acc.*') -> pd.DataFrame:
    '''
    Keeps only the acceleration columns (start with acc)
    inputs:
        df: dataframe, renamed columns raw imu file data
        regex: string, a regular expression 'acc.*' by default
    '''
    return df.filter(regex=regex)


def rolling_average(vector: np.array, window_width: Optional[int] = 151) -> np.array:
    '''
    Calculate the rolling average of a vector
    input:
        vector: numpy array, input to calculate the moving average
        window_with: integer, defines how many units in the vector to average together 151 items by default
    '''
    cumsum_vec = np.cumsum(np.insert(vector, 0, 0))
    ma_vec = (cumsum_vec[window_width:] -
              cumsum_vec[:-window_width]) / window_width
    return ma_vec


def calculate_vector_magnitude(df: pd.DataFrame,
                               x_vector: Optional[str] = 'acc_x',
                               y_vector: Optional[str] = 'acc_y',
                               z_vector: Optional[str] = 'acc_z') -> np.array:
    '''
    Calculate the magnitude of a 3D vector (x, y, z axis)
    The square root the sum of the squared vector values
    sqrt(Ax^2 + Ay^2 + Az^2)
    input:
        df: df: dataframe, renamed columns raw imu file data
        x_vector: string, x axis column name 'acc_x' by default
        y_vector: string, y axis column name 'acc_y' by default
        z_vector: string, z axis column name 'acc_z' by default    
    '''

    # square the vectors
    x_squared = df[x_vector] ** 2
    y_squared = df[y_vector] ** 2
    z_squared = df[z_vector] ** 2

    # square root the sum of squared values (magnitude)
    return np.sqrt(sum([x_squared + y_squared + z_squared]))


def identify_maximum_acceleration(df: pd.DataFrame,
                                  numeric_column: Optional[str] = 'acc_x') -> pd.DataFrame:
    '''
    Slices a dataframe to just 100 observations before, and after the maximum value of a numeric value
    input:
        df: dataframe, renamed columns raw imu file data
        numeric_column: string, numeric values column name 'acc_x' by default
    '''
    # find the index for the maximum value in the numeric column
    maximum_idx = df[numeric_column].idxmax()

    # capture 100 frames before and after the maximum value
    maximum = df.loc[maximum_idx - 100: maximum_idx + 100]
    return maximum


def intergrate_column(df: pd.DataFrame,
                      columns_to_integrate: Optional[List[str]] = ['acc_x', 'acc_y', 'acc_z'],
                      integrated_columns_names: Optional[List[str]] = ['veloc_x', 'veloc_y', 'veloc_z'],
                      x_column: Optional[str] = 'time') -> pd.DataFrame:
    '''
    Intergrate using the trapezoidal rule the acceleration of the wrist to produce the velocity
    input:
        df: dataframe, renamed columns raw imu file data
        columns_to_integrate: array, list containing columns to integrate
        integrated_columns_names: array, list containing names for the new integrated columns
        x_column: string, the x axis column name to produce the acceleration function
    '''
    assert len(integrated_columns_names) == len(columns_to_integrate), \
        "The new columns name array should be the same length as the columns to integerate array"

    for i in range(len(integrated_columns_names)):
        df[integrated_columns_names[i]] = np.append(
            0.0, cumtrapz(df[columns_to_integrate[i]], x=df[x_column]))
    return df


def scale_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns a standard scaled version of the input dataframe.
    column wise operation; subtract the mean and divide the standard deviation.
    input:
        df: dataframe, renamed columns raw imu file data
    '''
    # refrence column names
    column_names = df.columns.tolist()

    # instanciate scaler and fit to the dataframe
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    # create a new dataframe with the scaled features
    scaled_df = pd.DataFrame(
        scaled_features, columns=column_names, index=df.index)
    return scaled_df


def process_data(dictionary_df: pd.DataFrame, path_to_data: str) -> None:
    filenames = dictionary_df['Filename'].to_numpy()
    speeds = dictionary_df['Speed'].to_numpy()
    first_file = True

    try:
        for i, filename in enumerate(filenames):
            path = f'{path_to_data}{filename}.txt'
            data = pd.read_csv(path)

            # rename column names
            proccessed_data = rename_observation_columns(data)

            # calculate the velocities
            proccessed_data = intergrate_column(proccessed_data)

            # drop gyroscope data
            gryo_mask = proccessed_data.columns.str.startswith('gyro')
            gyro_columns = proccessed_data.columns[gryo_mask]
            proccessed_data = proccessed_data.drop(gyro_columns, axis=1)

            # smooth the imu readings by calculating the rolling average
            proccessed_data = proccessed_data.apply(
                lambda column: rolling_average(column.to_numpy()))

            # find the maximum value on x axis acceleration and slice 100 frames before and after that point
            proccessed_data = identify_maximum_acceleration(
                proccessed_data).reset_index(drop=True)

            # calculate acceleration magnitude
            proccessed_data['acceleration_magnitude'] = calculate_vector_magnitude(
                proccessed_data)

            # calculate acceleration magnitude
            proccessed_data['velocity_magnitude'] = calculate_vector_magnitude(proccessed_data,
                                                                               'veloc_x',
                                                                               'veloc_y',
                                                                               'veloc_z')

            # keep only magnitude columns
            mag_mask = proccessed_data.columns.str.endswith('magnitude')
            magnitude_columns = proccessed_data.columns[mag_mask]
            proccessed_data = proccessed_data.loc[:, magnitude_columns]

            # create a standard scaled version of the data
            proccessed_data = scale_dataframe(proccessed_data)

            # rehsape the values into a single row and re-create the dataframe
            column_names = [
                f'acceleration_{i}' if i < 201 else f'velocity_{i-201}' for i in range(len(proccessed_data) * 2)]
            flattened_values = proccessed_data.to_numpy().flatten().reshape(1, -1)

            # reshpe the dataframe into a single row
            proccessed_data = pd.DataFrame(
                flattened_values, columns=column_names)
            proccessed_data['speed'] = speeds[i]

            # append the row data into a new document
            proccessed_data.to_csv(
                'magnitudes.csv', mode='a', index=False, header=first_file)
            first_file = False

    except Exception as e:
        print(str(e))

    return

# Modeling Functions


def RMSE(actual: Union[List, int, float], predictions: Union[List, int, float]) -> float:
    '''
    Calculates the root mean squared errors for a set of predictions
    input:
        actual: the ground truth
        preidctions: the model predictions
    '''
    MSE = mean_squared_error(actual, predictions)
    return round(np.sqrt(MSE), 2)

def plot_feature_importance(model: BaseEstimator, 
                            feature_names: Union[pd.Series, np.array, List], n: Optional[int] = 12) -> None:
    
    ''' 
    Receives an sklearn type model and creates feature importances graph
    Note: Not all sklearn model object have this feature
    inputs:
        model: sklearn model object, or any other model with the feature_importances_ attribute
        feature_names: array, containing the names of all features the model was trained on
        n: integer, the number of features to include in the graph, 12 by default
    
    '''
    # extract the feature importances
    importances = model.feature_importances_    
    
    # combine the features importance and column names into a matrix 
    feature_matrix = np.array([importances, feature_names])
    
    # convert from two rows 'n' columns to 'n' rows two columns matrix and sort
    feature_matrix = feature_matrix.transpose()
    feature_matrix = feature_matrix[feature_matrix[:, 0].argsort()][::-1]
    
    # separate and limit the features and name
    importances = feature_matrix[:, 0][:n]
    names = feature_matrix[:, 1][:n]

    # plot the features
    plt.figure(figsize=(14, 10))
    plt.barh(names, importances, align='center')
    
    # adjust y ticks, add labels and titles
    plt.yticks(name[::-1], names)
    plt.xticks(rotation=45)
    plt.title('Feature Importances', fontsize=18)
    plt.xlabel('Importance', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    return

