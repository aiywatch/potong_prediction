## Import Libraries & Data
import datetime
import arrow
import pandas as pd
import numpy as np

import connection




class PotongCleaning:
    NUMBER_OF_DAYS = 21
    
    LINE_VID_MAP = {'1': ["359739072722465"],
                  '2': ["359739072730088"],
                  '2a': ["352648061891537"],
                  '3': ["358901049778803"]}
    
    def __init__(self):
        self.BUS_DATA_ADAPTER = connection.connect_mongo_bus_status()


    def _query_bus_data_from_mongo(self, bus_vid):
        """ bus_vid = array of vid of each bus line """
        start_date = datetime.date.today() - datetime.timedelta(days=(self.NUMBER_OF_DAYS))
        start_date = arrow.get(start_date, 'Asia/Bangkok').datetime
        end_date = datetime.date.today() - datetime.timedelta(days=3)
        end_date = arrow.get(end_date, 'Asia/Bangkok').datetime
        
        print('Connecting to mongodb ...')
        bus_data_list = []
        for bus_data in self.BUS_DATA_ADAPTER.find({'vehicle_id': {'$in': bus_vid}, 'gps_timestamp': {'$gte': start_date, '$lte': end_date}}):
            bus_data_list += [bus_data]
            print(bus_data['vehicle_id'], bus_data['gps_timestamp'])
        
        bus_data_df = pd.DataFrame(bus_data_list)
        
        return bus_data_df
    
    def clean_data(self, bus_line):
        
        def _clean_status_and_trip_id(data):
            """ Clean incorrect inbound/outbound and add trip_id for the groups of 
                contiguous points with the same status """ 
            
            def _get_linear_ref(dt):
                return dt['linear_outbound'] if (dt['status'] == 'outbound') else dt['linear_inbound']
            
            data['linear_ref'] = data.apply(_get_linear_ref, axis=1).astype(float)
            
            data['linear_inbound'] = data['linear_inbound'].astype(float)
            data['linear_outbound'] = data['linear_outbound'].astype(float)
            
            data.sort_values(['vehicle_id', 'gps_timestamp'], inplace=True)
            
            status = []
            trip_id = []
            count_trip = 0
            last_status = ""
            
            for i, point in data.iterrows():
                lower = i-5 if i-5>=0 else 0
                upper = i+6 if i+6<=data.shape[0] else data.shape[0]
                
                status += [data.iloc[lower:upper, :]['status'].value_counts().idxmax()]
                
                
                prev_index = i-1 if i-1 >= 0 else 0
                if((status[-1] != last_status)  | 
                        (point['linear_ref'] - data.loc[prev_index, 'linear_ref'] > 0.6) |
                        (abs((point['gps_timestamp'] - data.loc[prev_index, 'gps_timestamp']).seconds) > 500) ):
                    last_status = point['status']
                    count_trip += 1
                trip_id += [count_trip]
                
            
            data['status'] = status
            data['trip_id'] = trip_id
            
            return data    
    
        def _include_next_point(data, n):
            """ Create new dataframe for 2 GPS points combined (current point 
            and next point, which provides time to next point to predict) """
            def _get_distance(df):
                if df['status'] == 'inbound':
                    return df['next_lin_in'] - df['linear_inbound']
                if df['status'] == 'outbound':
                    return df['next_lin_out'] - df['linear_outbound']
            
            def _get_linear_ref(dt):
                return dt['linear_inbound'] if dt['status'] == 'inbound' else dt['linear_outbound']
                
            temp_data = data.copy()
            temp_data['next_time'] = temp_data['gps_timestamp'].shift(-n)
            temp_data['next_lin_in'] = temp_data['linear_inbound'].shift(-n)
            temp_data['next_lin_out'] = temp_data['linear_outbound'].shift(-n)
            temp_data['next_linear_ref'] = temp_data['linear_ref'].shift(-n)
            temp_data['next_status'] = temp_data['status'].shift(-n)
            temp_data['next_trip_id'] = temp_data['trip_id'].shift(-n)
            
            temp_data['next_index'] = temp_data['index'].shift(-n)
            
            temp_data = temp_data[temp_data['trip_id']==temp_data['next_trip_id']]
            
            if temp_data.shape[0] == 0:
                return None
            
            temp_data['time_to_next'] = temp_data['next_time'] - temp_data['gps_timestamp']
            temp_data = temp_data[temp_data['status'] == temp_data['next_status']]
            temp_data['distance_to_next'] = temp_data.apply(_get_distance, axis=1)
            
            
            
            temp_data['hour'] = temp_data['gps_timestamp'].apply(lambda dt: dt.hour)
            temp_data['day_of_week'] = temp_data['gps_timestamp'].apply(lambda dt: dt.weekday())
            temp_data['linear_ref'] = temp_data.apply(_get_linear_ref, axis=1)
            
            
            return temp_data
        
    #    bus_line = '1'
        
        raw_data = self._query_bus_data_from_mongo(self.LINE_VID_MAP[bus_line])
        print("data loaded")
        if raw_data.shape[0] < 10:
            print('Error from Data base')
            return None, None
        
        data = raw_data.copy()
        
        
        data = data[["gps_timestamp", "speed", "vehicle_id", "linear_inbound", 
                 "linear_outbound", "longitude", "latitude", "bus_line", "status"]]
        
        data = data[(data['status'] == 'inbound') | (data['status'] == 'outbound')]
        
        data = data.iloc[::6, :]
        
    #    data['linear_inbound'].astype(float, inplace=True)
        
        
        index = data.index.values
        data = data.reset_index(drop=True)
        data['index'] = index
    
        data['gps_timestamp'] = pd.to_datetime(data['gps_timestamp'])
    
        print("Start cleaning status and adding trip id")
        data = _clean_status_and_trip_id(data)
        print("status cleaned and trip id added")
        
        
        # Create data for training by combining 2 point
        print("combining data...")
        made_data = pd.DataFrame()
        for t in range(3, 100):
            print("combining every", t, "point")
            temp = _include_next_point(data, t)
            if temp is not None:
                made_data = pd.concat([made_data, temp])
        
        made_data = made_data[made_data['distance_to_next'] >= 0]
        made_data['time_to_next'] = made_data['time_to_next'].dt.seconds
    
        return made_data, data
    
    
    def run(self, bus_line):
        cleaned_data, data = self.clean_data(bus_line)
        if cleaned_data is not None:
            print("data cleaned!")
            cleaned_data.to_csv("data/cleaned_potong{}.csv.gz".format(bus_line), compression='gzip', index=False)
            print("saved")
        else:
            print("No data to be saved!")

class PotongModeling:
    
    SAVED_MODEL_PATH = 'data/saved_model/'

#    def __init__():

    def _get_modellers(self, bus_line, model_type, load_cached_data=False):
        """ Train Keras Deep Learning Model """
        
        def _get_X_y(made_data, model_type):
            """ Get X and y for Deep learning Model """
        
            if model_type == 'time':
                X_cols = ['day_of_week', 'status', 'hour', 'distance_to_next',
                        'speed', 'linear_ref']
                y_col = ['time_to_next']
                
            elif model_type == 'location':
                X_cols = ['day_of_week', 'status', 'hour',
                        'speed', 'linear_ref', 'time_to_next']
                y_col = ['next_linear_ref']
            
            X = made_data[X_cols].values
            y = made_data[y_col].values
            
            return X, y
        
        ## Import cleaned data, the result from cleaning_data.py
        if load_cached_data:
            
            DATA_PATH = "data/cleaned_potong{}.csv.gz".format(bus_line)
            made_data = pd.read_csv(DATA_PATH)
            print('loaded from cache')
        else:
            potong_cleaning = PotongCleaning()
            made_data, data = potong_cleaning.clean_data(bus_line)
        
        
        if made_data is None:
            return [None, None, None, None, None, None]
        
        made_data = made_data.dropna()
        
        ## Get X and y
        X, y = _get_X_y(made_data, model_type)
        
        ## Encode data
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
        labelencoder = LabelEncoder()
        X[:, 1] = labelencoder.fit_transform(X[:, 1])
        
        onehotencoder = OneHotEncoder(categorical_features=[0])
        X = onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]
        
        ## Split Train/ Test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        
        ## Scale Data
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        
        #import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from keras import regularizers
        
        feature_num = X.shape[1]
        
        regressor = Sequential()
        
        regressor.add(Dense(output_dim=30, init='normal', activation='relu', input_dim=feature_num,
                   kernel_regularizer=regularizers.l2(0.010)))
        regressor.add(Dense(output_dim=30, init='normal', activation='relu',
                   kernel_regularizer=regularizers.l2(0.010)))
        regressor.add(Dense(output_dim=30, init='normal', activation='relu',
                   kernel_regularizer=regularizers.l2(0.010)))
        regressor.add(Dense(output_dim=1, init='normal'))
        
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressor.fit(X_train, y_train, batch_size=32, nb_epoch=100, validation_data=(X_test, y_test))
    
        return [regressor, labelencoder, onehotencoder, sc_X, X_test, y_test]
    
    def _save_model(self, modellers, xy, bus_line, model_type):
        """ Save models to disk
            1. regressor - Deep learning structure to model.json & weight to model.h5
            2. encoders to encoders.pkl
            3. X_test, y_test to Xy.pkl for testing the model"""
            
        from sklearn.externals import joblib
        
        [regressor, labelencoder, onehotencoder, sc] = modellers
        [X_test, y_test] = xy
        
        # serialize model to JSON
        model_json = regressor.to_json()
        with open("{}{}/{}/model.json".format(self.SAVED_MODEL_PATH, model_type, bus_line), "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        regressor.save_weights("{}{}/{}/model.h5".format(self.SAVED_MODEL_PATH, model_type, bus_line))
        print("Saved model to disk")
        
        joblib.dump([labelencoder, onehotencoder, sc], "{}{}/{}/encoders.pkl".format(self.SAVED_MODEL_PATH, model_type, bus_line))
        joblib.dump([X_test, y_test], "{}{}/{}/Xy.pkl".format(self.SAVED_MODEL_PATH, model_type, bus_line))
    
    def run(self, bus_line, model_type, load_cached_data=False):
        
        [regressor, labelencoder, onehotencoder, sc_X, X_test, y_test] = self._get_modellers(bus_line, model_type, load_cached_data)
    
        if regressor is None:
            return None
        self._save_model([regressor, labelencoder, onehotencoder, sc_X], [X_test, y_test], bus_line, model_type)


#potong_modeling = PotongModeling()
#potong_modeling.run('1', 'time', load_cached_data=True)
#potong_modeling.run('2', 'time', load_cached_data=True)
#potong_modeling.run('2a', 'time', load_cached_data=True)
#run('3', 'time')

potong_modeling = PotongModeling()
potong_modeling.run('1', 'location', load_cached_data=True)
potong_modeling.run('2', 'location', load_cached_data=True)
potong_modeling.run('2a', 'location', load_cached_data=True)


#potong_clean = PotongCleaning()
#potong_clean.run('1')
#potong_clean.run('2')
#potong_clean.run('2a')





