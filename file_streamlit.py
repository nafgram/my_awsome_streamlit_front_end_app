import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import array
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


header = st.container()
project_text = st.container()
map_figure = st.container()
dataset = st.container()
features = st.container()
plots = st.container()
data_preparation_model = st.container()
predictionResult = st.container()
performanceResults = st.container()

st.markdown(
    """
    <style>
    .main{
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html = True
 )

@st.cache
def get_data(filename):
    data = pd.read_csv('DATA/merged_wtr-level_alerts_precip_hum_temp_press_wind_with-NaN.csv')
    data = data.set_index('date')
    return data


with header:
    st.title('Omdena Serbia Chapter: Reducing Flood Risks in Belgrade Area through AI Solutions')

with project_text:
    image = Image.open('Omdena_Serbia_chapter.png')
    new_image = image.resize((350, 400))
    st.image(new_image)

    st.markdown(' The goals:')
    st.markdown('The first goal was the creation of a supervised ML model, based on historical weather\
    and hydrological data for Danube, Sava, and other small rivers in the Belgrade area,\
    to allow the prediction of possible floods in the future.\
    The second goal was to deploy the ML model.')
    st.markdown('We compiled hydrological data for the Belgrade area and developed a\
    ML model to predict the water-level in Belgrade a day ahead.')
    st.markdown ('The ML model was further integrated into  the web as an API for flood prediction.')

with map_figure:
    st.subheader('River map of Serbia')
    st.text('image credit: Wikipedia')
    
    image = Image.open('Serbia_rivers.png')
    new_image = image.resize((500, 700))
    st.image(new_image)

with dataset:
    st.subheader('Hydro-meteorological data at Belgrade and Sava gauge stations')
    st.markdown('For our project, we looked at water-levels, precipitation, and other hydrometeorological\
                parameters and focussed on time fragments of recent floods from 2010 to 2020.')
    
    data = get_data('DATA/merged_wtr-level_alerts_precip_hum_temp_press_wind_with-NaN.csv')

    start_date = st.date_input('Enter start date ', value=datetime(year=2010, month=1, day=1))
    starting_date = start_date.strftime("%Y-%m-%d")
    
    end_date = st.date_input('Enter end date', value=datetime(year=2020, month=12, day=31))
    ending_date = end_date.strftime("%Y-%m-%d") 
  
    #input_parameter = st.selectbox('Which feature would you like to predict?', 
     #                              options = ['belgrade_water_level_cm', 'belgrade_precipitation_mm', 
      #                                           'belgrade_humidity_pct', 'belgrade_pressure_hg', 'belgrade_temperature_c',
       #                                            'belgrade_windspeed_kph'])
    data_selected_period= data[str(starting_date): str(ending_date)]
    #data_file = data_selected_period[input_parameter]
    data_file = data_selected_period['belgrade_water_level_cm']
    st.write(data_file.head(5))
        
with features:
    st.subheader('Feature selection')
   
    input_feature = st.selectbox('Which feature would you like to plot?', 
                                      options = ['belgrade_water_level_cm', 'belgrade_precipitation_mm', 
                                                 'belgrade_humidity_pct', 'belgrade_pressure_hg', 'belgrade_temperature_c',
                                                   'belgrade_windspeed_kph'])
    

with plots:
    st.subheader('Plot of selected hydrogeological parameter')
    st.line_chart(pd.DataFrame(data_selected_period[input_feature]))

with data_preparation_model:  
    st.subheader('Using the model')

    #n_features = sel_col.selectbox('How many features do you want to use for the prediction',
     #                              options = [1, 2, 3, 4, 5, 6])
    #n_steps = st.selectbox('how many time lags would you like to use?',
    #                            options = [1, 2, 3, 4, 5, 6])

    # For demonstration purposes we set n_step, the time steps or lags, to 4 and n_features to 1
    n_features = 1
    n_steps=4
    st.markdown('The model in this API is an LSTM model trained using 4 water-level lags.')
    st.markdown(' The performances of the LSTM model on training and test data are: ')
    st.text('Train RMSE: 6.01, Test RMSE: 5.83, Train MAE: 4.56, Test MAE: 4.31, r2 score: 1.00')
# loading the model with pickle
# We will rather import the model that was already trained externally as LSTM is very time consuming
###########################################################################

loaded_model = pickle.load(open('trained_LSTM_model_NO_scaling.sav','rb'))

with predictionResult:
    index_num = 100000000
   
    st.subheader('Prediction of water-level in Belgrade')
    st.markdown('You can test prediction by using water-level lags from the following test set.\
                The test file was not used to train the LSTM model.')
   
    sel_col, displ_col = st.columns(2)
    # split into train and test sets
    train_size = int(len(data_file) * 0.75)
    test_size = len(data_file) - train_size
    train_file, test_file = data_file[0:train_size], data_file[train_size : len(data_file)]

    st.write('Length of training set: ',len(train_file), 'Length test set: ', len(test_file))

# split a univariate sequence into samples
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
 # find the end of this pattern
            end_ix = i + n_steps
 # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
 # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        return array(X), array(y)

    trainX, trainY = split_sequence(train_file, n_steps)
    testX, testY   = split_sequence(test_file, n_steps)
    
    testX_df = pd.DataFrame (testX, columns = ['lag{}'.format(i+1) for i in range(n_steps)])
    testY_df = pd.DataFrame(testY)
    testY_df.columns = ['Values to be predicted']
    test_total = pd.concat([testX_df, testY_df], axis=1)
    st.write(test_total)
    st.markdown('Please do provide 4 lags you can select from the test file above',)

    def get_number_inputs():
        nums = []
        #num_count = st.number_input(
         #                           min_value=1, max_value=100, value=1, step=1, 
         #                           key='num_count')
        num_count = 4
        for i in range(num_count):
            num = st.number_input(f"Input number {i+1}")
            #st.write(type(num))
            nums.append(num)
        return nums


#sel_col.text('Please enter the water-level lags you would like to use for prediction: ')
sel_col.text('4 lags are used for water-level')
nums = get_number_inputs()
nums_array= np.array(nums)
x_input  = nums
#st.write('nums', nums)

x1, x2, x3, x4 = nums
for i in range(len(test_total) - 3):
    if (test_total['lag1'][i] == x1) & (test_total['lag2'][i] == x2) & (test_total['lag3'][i] == x3)\
        & (test_total['lag4'][i] == x4):
        index_num = i

if index_num == 100000000:
    st.write('Please do not forget to enter the 4 lags')
else:  
    st.write('row number corresponding to test data: ', index_num)

    st.subheader('The real water level value is: ')
    y_real = test_total.iloc[index_num]['Values to be predicted']
    st.write(y_real)

    x_input = nums_array.reshape((1, n_steps, n_features))
    st.subheader('The predicted value is: ')
    y_predicted = loaded_model.predict(x_input)
    st.write(y_predicted)

# Performance results
    st.subheader('The mean absolute value is: ')
    mae = abs(y_real - y_predicted)
    st.write(mae)
