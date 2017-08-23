## Import Libraries & Data
import datetime
import arrow
import pandas as pd
import numpy as np

import connection

NUMBER_OF_DAYS = 14
BUS_DATA_ADAPTER = connection.connect_mongo_bus_status()

LINE_VID_MAP = {'1': ["359739072722465"],
              '2': ["359739072730088"],
              '2a': ["352648061891537"],
              '3': ["358901049778803"]}



def query_bus_data_from_mongo(bus_vid):
    """ bus_vid = array of vid of each bus line """
    start_date = datetime.date.today() - datetime.timedelta(days=(NUMBER_OF_DAYS))
    start_date = arrow.get(start_date, 'Asia/Bangkok').datetime
    end_date = datetime.date.today() - datetime.timedelta(days=3)
    end_date = arrow.get(end_date, 'Asia/Bangkok').datetime
    
    print('Connecting to mongodb ...')
    bus_data_list = []
    for bus_data in BUS_DATA_ADAPTER.find({'vehicle_id': {'$in': bus_vid}, 'gps_timestamp': {'$gte': start_date, '$lte': end_date}}):
        bus_data_list += [bus_data]
        print(bus_data['vehicle_id'], bus_data['gps_timestamp'])
    
    bus_data_df = pd.DataFrame(bus_data_list)
    
    return bus_data_df

def clean_data(bus_line):
    
    def clean_status_and_trip_id(data):
        """ Clean incorrect inbound/outbound and add trip_id for the groups of 
            contiguous points with the same status """ 
        
        def get_linear_ref(dt):
            return dt['linear_outbound'] if (dt['status'] == 'outbound') else dt['linear_inbound']
        
        data['linear_ref'] = data.apply(get_linear_ref, axis=1).astype(float)
        
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

    def include_next_point(data, n):
        """ Create new dataframe for 2 GPS points combined (current point 
        and next point, which provides time to next point to predict) """
        def get_distance(df):
            if df['status'] == 'inbound':
                return df['next_lin_in'] - df['linear_inbound']
            if df['status'] == 'outbound':
                return df['next_lin_out'] - df['linear_outbound']
        
        def get_linear_ref(dt):
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
        temp_data['distance_to_next'] = temp_data.apply(get_distance, axis=1)
        
        
        
        temp_data['hour'] = temp_data['gps_timestamp'].apply(lambda dt: dt.hour)
        temp_data['day_of_week'] = temp_data['gps_timestamp'].apply(lambda dt: dt.weekday())
        temp_data['linear_ref'] = temp_data.apply(get_linear_ref, axis=1)
        
        
        return temp_data
    
#    bus_line = '1'
    
    raw_data = query_bus_data_from_mongo(LINE_VID_MAP[bus_line])
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
    data = clean_status_and_trip_id(data)
    print("status cleaned and trip id added")
    
    
    # Create data for training by combining 2 point
    print("combining data...")
    made_data = pd.DataFrame()
    for t in range(3, 100):
        print("combining every", t, "point")
        temp = include_next_point(data, t)
        if temp is not None:
            made_data = pd.concat([made_data, temp])
    
    made_data = made_data[made_data['distance_to_next'] >= 0]
    made_data['time_to_next'] = made_data['time_to_next'].dt.seconds

    return made_data, data


def run(bus_line):
    cleaned_data, data = clean_data(bus_line)
    if cleaned_data is not None:
        print("data cleaned!")
        cleaned_data.to_csv("data/cleaned_potong{}.csv.gz".format(bus_line), compression='gzip', index=False)
        print("saved")
    else:
        print("No data to be saved!")



#run('1')
#run('2')
#run('2a')
#run('3')



