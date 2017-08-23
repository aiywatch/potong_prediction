from sklearn.externals import joblib
import pandas as pd
import datetime
import requests
import math
from keras.models import model_from_json
#from geopy.distance import vincenty

import connection




class PotongPrediction:
    """ Parent class of both location prediction class and time prediction class
        collecting common functions used by both classes """
        
    START_END_TERMINALS = {'1': ['ห้างสรรพสินค้าบิ๊กซี', 'วิทยาลัยอาชีวศึกษาภูเก็ต'],
                          '2': ['วิทยาลัยอาชีวศึกษาภูเก็ต', 'ตลาดใหม่มุมเมือง'],
                          '2a': ['ขนส่ง 1', 'ขนส่ง 2'],
                          '3': ['สะพานหิน', 'เกาะสิเหร่']}
    
    BUS_LINES = ['1', '2', '2a', '3']
    
    def __init__(self):
        self.linref_latlon_cursor = connection.connect_postgres_linref_latlon()


    def _get_user_route_data(self, usr_lat, usr_lon, bus_line, direction):
        
        if(bus_line == '2a'):
            bus_line = '2_p'
        bus_line = "pothong_" + bus_line
        
        if(direction == 'in'):
            direction = 'inbound'
        elif(direction == 'out'):
            direction = 'outbound'
            
        self.linref_latlon_cursor.execute("""SELECT bus_line, direction, ST_Y(ST_ClosestPoint(geometry, ST_SetSRID('POINT({0} {1})', 4326))) 
        as lat, ST_X(ST_ClosestPoint(geometry, ST_SetSRID('POINT({0} {1})', 4326))) 
        as lng, ST_Line_Locate_Point(geometry,ST_SetSRID(ST_Point({0}, {1}),4326)) 
        linear_ref FROM busbmta.terminal_and_route where bus_line = '{2}' and direction = '{3}' 
        and geometry_name = 'route'""".format(usr_lon, usr_lat, bus_line, direction))
    
        rows = self.linref_latlon_cursor.fetchall()
        return rows[0]

    def _import_model(self, bus_line, model_type):
        MODEL_PATH = "data/saved_model"
        [labelencoder, onehotencoder, sc] = joblib.load("{}/{}/{}/encoders.pkl".format(MODEL_PATH, model_type, bus_line))
        
        # load json and create model
        json_file = open("{}/{}/{}/model.json".format(MODEL_PATH, model_type, bus_line), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        regressor = model_from_json(loaded_model_json)
        # load weights into new model
        regressor.load_weights("{}/{}/{}/model.h5".format(MODEL_PATH, model_type, bus_line))
        print("Loaded model from disk")
    
        return [regressor, labelencoder, onehotencoder, sc]


    def _extract_bus_info(self, bus):
        bus_data = pd.Series()
        bus_data['vehicle_id'] = bus['vehicle_id']
        bus_data['timestamp'] = bus['info']['gps_timestamp']
        bus_data['name'] = bus['name']
        bus_data['linear_ref'] = bus['checkin_data']['route_linear_ref']
        bus_data['speed'] = bus['info']['speed']
        bus_data['direction'] = bus['info']['direction']
        bus_data['status'] = bus['status']
        [bus_data['lon'], bus_data['lat']] = bus['info']['coords']
        bus_data['license_id'] = bus['desc']['license_id']
        
        bus_data['prev_stop'] = bus['checkin_data']['prev_stop']['name']
        bus_data['next_stop'] = bus['checkin_data']['next_stop']['name']
    
        return bus_data

    def _encode_data(self, data_point, labelencoder, onehotencoder, sc):
        if(data_point['direction'] == 'in'):
            data_point['direction'] = 'inbound'
        elif(data_point['direction'] == 'out'):
            data_point['direction'] = 'outbound'
        new_data_point = data_point.copy()
        
        new_data_point[1] = labelencoder.transform([new_data_point[1]])[0]
        
        new_data_point = onehotencoder.transform([new_data_point]).toarray()
        new_data_point = new_data_point[0, 1:]
        
        new_data_point = sc.transform([new_data_point])
        return new_data_point





    def _clean_data_time(self, data_point, usr_linear_ref):
        new_data_point = data_point.copy()
        new_data_point['timestamp'] = pd.to_datetime(new_data_point['timestamp'])
        new_data_point['day_of_week'] = new_data_point['timestamp'].weekday()
        new_data_point['hour'] = new_data_point['timestamp'].hour
        new_data_point['direction'] = 'outbound' if new_data_point['direction'] == 'out' else 'inbound'
       
        
        new_data_point['distance_to_next'] = usr_linear_ref - new_data_point['linear_ref']
        
    
        new_data_point = new_data_point[['day_of_week', 'direction', 'hour', 'distance_to_next',
                                         'speed', 'linear_ref']]
        return new_data_point
    
    def _clean_data_location(self, data_point, time):
        new_data_point = data_point.copy()
        new_data_point['timestamp'] = pd.to_datetime(new_data_point['timestamp'])
        new_data_point['day_of_week'] = new_data_point['timestamp'].weekday()
        new_data_point['hour'] = new_data_point['timestamp'].hour
        new_data_point['time_to_next'] = (time - new_data_point['timestamp']).total_seconds()
        new_data_point['last_point_lat'] = new_data_point['lat']
        new_data_point['last_point_lon'] = new_data_point['lon']
        new_data_point['location_zone'] = math.floor(new_data_point['linear_ref'] * 10000)
    
        new_data_point = new_data_point[['day_of_week', 'direction', 'hour', 
                                         'speed', 'linear_ref', 'time_to_next']]
        return new_data_point

class PotongLocation(PotongPrediction):
    """ Location prediction """

    def _get_lastest_gps(self, bus_vehicle_id):
        
        data = requests.get('https://api.traffy.xyz/vehicle/?vehicle_id='+str(bus_vehicle_id)).json()
        bus = data['results']
        if(bus):
            return bus[0]
        return None

    def predict_location(self, bus_line, bus_vehicle_id):
        bus_line = str(bus_line)
        if(bus_line not in self.BUS_LINES):
            return 'This bus line is not available!'
            
        bus = self._get_lastest_gps(bus_vehicle_id)
        
        if(not bus):
            return 'This bus is not available!'
            
        bus_data = self._extract_bus_info(bus)

        
    
        time_now = pd.to_datetime(datetime.datetime.utcnow())
        [regressor, labelencoder, onehotencoder, sc] = self._import_model(bus_line, 'location')
        
        cleaned_bus_data = self._clean_data_location(bus_data, time_now)
        encoded_bus_data = self._encode_data(cleaned_bus_data, labelencoder, onehotencoder, sc)
        
        
        predicted_location = regressor.predict([encoded_bus_data])
    
        output = {'predicted_linear_ref': predicted_location[0],
                        'last_point_data': {
                            'last_timestamp': bus_data['timestamp'],
                            'timestamp_now': time_now,
                            'time_to_next': cleaned_bus_data['time_to_next'],
                            'last_linear_ref': bus_data['linear_ref'],
                            'last_speed': bus_data['speed'],
                            'direction': bus_data['direction'],}
                        }
              
        return output


class PotongTime(PotongPrediction):
    """ Time Prediction """

    def predict_time(self, bus_line, bus_data, usr_linear_ref, usr_route_lat, usr_route_lon):
        """ 1. Import Deep learning Model from Disk
            2. Clean/Filter only required bus data and Encode the data
            3. Predict arrival time """
        
        
        time_now = pd.to_datetime(datetime.datetime.utcnow())
        
        [regressor, labelencoder, onehotencoder, sc] = self._import_model(bus_line, model_type='time')
        cleaned_bus_data = self._clean_data_time(bus_data, usr_linear_ref)
        encoded_bus_data = self._encode_data(cleaned_bus_data, labelencoder, onehotencoder, sc)
        
        predicted_time = regressor.predict([encoded_bus_data])
        
        output = {'predicted_arrival_time': float(predicted_time[0][0]),
                  'vehicle_id': bus_data['vehicle_id'],
                  'bus_line': bus_line,
                  'direction': bus_data['direction'],
                  'prev_stop': bus_data['prev_stop'],
                  'next_stop': bus_data['next_stop'],
                  'license_id': bus_data['license_id'],
                        'last_point_data': {
                            'user_linear_ref': usr_linear_ref,
                            'usr_route_lat_lon': [usr_route_lat, usr_route_lon],
                            'last_timestamp': bus_data['timestamp'],
                            'timestamp_now': str(time_now),
                            'distance_to_next': cleaned_bus_data['distance_to_next'],
                            'last_bus_linear_ref': bus_data['linear_ref'],
                            'last_bus_lat_lon': [bus_data['lat'], bus_data['lon']],
                            'last_speed': bus_data['speed'],
                            'bus_name': bus_data['name'],}
                        }
        return output


    def predict_time_per_line(self, bus_line, usr_lat, usr_lon, usr_dir):
        """ 1. Preprocessing user data(i.e. user's latitude/longitude and direction
            2. Getting User's linear ref
            3. Querying a matched bus(same bus line, direction, and havn't passed) from API
            4. Calling Machine learning function (predict_time)"""
                
        
        ## Getting Buses data in the requested bus line
        data = requests.get('https://api.traffy.xyz/vehicle/?line=potong-{}'.format(
                bus_line)).json()
        buses = data['results']
        
        if not buses:
            return None
        
        ## Extract raw bus JSON/dict data to Pandas DataFrame
        bus_df = pd.DataFrame(columns=['vehicle_id', 'timestamp', 'name', 'linear_ref', 
                                  'speed', 'direction', 'lat', 'lon', 'status'])
        for bus in buses:
            try:
                bus_info = self._extract_bus_info(bus)
                bus_df = bus_df.append(bus_info, ignore_index=True)
            except:
                print('Info error')
                pass
        
        
        _, _, usr_route_lat, usr_route_lon, usr_linear_ref = self._get_user_route_data(usr_lat, usr_lon, bus_line, usr_dir)
    
        
        
        ## Filtering buses
        running_buses = bus_df[(bus_df['status'] == usr_dir) & 
                               (bus_df['linear_ref'] < usr_linear_ref)
                              ].sort_values('linear_ref', ascending=False)
        
    #    print(running_buses)
        if running_buses.shape[0] != 0:
            closest_bus = running_buses.iloc[0, :]
            
            predict_result = self.predict_time(bus_line, closest_bus, usr_linear_ref, usr_route_lat, usr_route_lon)    
            return predict_result

    def request_time_prediction(self, bus_line, usr_lat, usr_lon, usr_dir):
        predict_result = self.predict_time_per_line(bus_line, usr_lat, usr_lon, usr_dir)
        if predict_result:
            return {'results': predict_result, 'status': True}
        else:
            return {'results': "", 'status': False}



    def request_time_prediction_all_line(self, usr_lat, usr_lon):
        print('requesting...')
        predict_times = []
        for bus_line in self.BUS_LINES:
            results = []
            for dir in ('in','out'):
                result = self.predict_time_per_line(bus_line, usr_lat, usr_lon, dir)
                if result is not None:
                    results += [result]
            if results:
                predict_times += [results]
            
        if len(predict_times) > 0:
            result = {'results': predict_times, 'start_end_names': self.START_END_TERMINALS,
                      'status': True}
            return result
        else: 
            result = {'results': [], 'start_end_names': self.START_END_TERMINALS,
                      'status': False}
            return result





potong_time = PotongTime()


bus_line = '1'
##usr_lon = 98.3663
##usr_lat = 7.89635
usr_lon = 98.401550
usr_lat = 7.862933

data_dict = potong_time.request_time_prediction_all_line(usr_lat, usr_lon)
#
import json
json.dumps(data_dict, ensure_ascii=False)
#json.dumps(data_dict)


usr_dir = 'out'
potong_time.request_time_prediction(bus_line, usr_lat, usr_lon, usr_dir)
#request_prediction_all_line(usr_lat, usr_lon)


potong_location = PotongLocation()

potong_location.predict_location('1', "059049183")


#bus_vehicle_id_1 = 359739072722465
#bus_vehicle_id_2 = 359739072730088
#bus_vehicle_id_2a = 352648061891537
#bus_vehicle_id_3 = 358901049778803
#
#
##usr_lat = 98.3663
##usr_lon = 7.89635
#
#usr_lat = 98.3633795
#usr_lon = 7.858667
#
##usr_lat = 98.395044
##usr_lon = 7.8677455
##
##usr_lat = 98.36688516666666
##usr_lon = 7.887917666666667
#
#cleaned_bus_data = predict_time(2, bus_vehicle_id_2, usr_lat, usr_lon, 'in')











