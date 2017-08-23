from sklearn.externals import joblib
import pandas as pd
import datetime
import requests
import math
from keras.models import model_from_json
#from geopy.distance import vincenty

import connection

cursor = connection.connect_postgres_linref_latlon()


BUS_LINES = ['1', '2', '2a', '3']


def get_user_route_data(usr_lat, usr_lon, bus_line, direction):
    
    if(bus_line == '2a'):
        bus_line = '2_p'
    bus_line = "pothong_" + bus_line
    
    if(direction == 'in'):
        direction = 'inbound'
    elif(direction == 'out'):
        direction = 'outbound'
        
    cursor.execute("""SELECT bus_line, direction, ST_Y(ST_ClosestPoint(geometry, ST_SetSRID('POINT({0} {1})', 4326))) 
    as lat, ST_X(ST_ClosestPoint(geometry, ST_SetSRID('POINT({0} {1})', 4326))) 
    as lng, ST_Line_Locate_Point(geometry,ST_SetSRID(ST_Point({0}, {1}),4326)) 
    linear_ref FROM busbmta.terminal_and_route where bus_line = '{2}' and direction = '{3}' 
    and geometry_name = 'route'""".format(usr_lon, usr_lat, bus_line, direction))

    rows = cursor.fetchall()
    return rows[0]

def import_model(bus_line):
    MODEL_PATH = "saved-model"
    [labelencoder, onehotencoder, sc] = joblib.load("{}/{}/encoders.pkl".format(MODEL_PATH, bus_line))
    
    # load json and create model
    json_file = open("{}/{}/model.json".format(MODEL_PATH, bus_line), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    regressor = model_from_json(loaded_model_json)
    # load weights into new model
    regressor.load_weights("{}/{}/model.h5".format(MODEL_PATH, bus_line))
    print("Loaded model from disk")

    return [regressor, labelencoder, onehotencoder, sc]






def extract_bus_info(bus):
    bus_data = pd.Series()
    bus_data['vehicle_id'] = bus['vehicle_id']
    bus_data['timestamp'] = bus['info']['gps_timestamp']
    bus_data['name'] = bus['name']
#    bus_data['trip_id'] 
    bus_data['linear_ref'] = bus['checkin_data']['route_linear_ref']
    bus_data['speed'] = bus['info']['speed']
    bus_data['direction'] = bus['info']['direction']
    bus_data['status'] = bus['status']
    [bus_data['lon'], bus_data['lat']] = bus['info']['coords']

    return bus_data

def clean_data(data_point, time):
    new_data_point = data_point.copy()
    new_data_point['timestamp'] = pd.to_datetime(new_data_point['timestamp'])
    new_data_point['day_of_week'] = new_data_point['timestamp'].weekday()
    new_data_point['hour'] = new_data_point['timestamp'].hour
    new_data_point['time_to_next'] = (time - new_data_point['timestamp']).total_seconds()
#    new_data_point['last_point_location'] = new_data_point['linear_ref']
    new_data_point['last_point_lat'] = new_data_point['lat']
    new_data_point['last_point_lon'] = new_data_point['lon']
#    data_point.timestamp + datetime.timedelta(0,5)
    new_data_point['location_zone'] = math.floor(new_data_point['linear_ref'] * 10000)

    new_data_point = new_data_point[['day_of_week', 'direction', 'hour', 
                                     'speed', 'linear_ref', 'time_to_next']]
    return new_data_point

def encode_data(data_point, labelencoder, onehotencoder, sc):
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

def get_lastest_gps(bus_vehicle_id):
    
    data = requests.get('https://api.traffy.xyz/vehicle/?vehicle_id='+str(bus_vehicle_id)).json()
    bus = data['results']
    if(bus):
        return bus[0]
    return None

def predict_location(bus_line, bus_vehicle_id):
#    print(bus_line,bus_id)
    bus_line = str(bus_line)
    if(bus_line not in BUS_LINES):
        return 'This bus line is not available!'
        
#    print(pd.to_datetime(datetime.datetime.utcnow()))
    bus = get_lastest_gps(bus_vehicle_id)
    
    if(not bus):
        return 'This bus is not available!'
        
#    [regressor, labelencoder, onehotencoder] = import_model('potong-' + bus_line)
#
    bus_data = extract_bus_info(bus)
#    time = pd.to_datetime(datetime.datetime.utcnow())
#    cleaned_bus_data = clean_data(bus_data, time)
#    encoded_bus_data = encode_data(cleaned_bus_data, labelencoder, onehotencoder)
    

    time_now = pd.to_datetime(datetime.datetime.utcnow())
    [regressor, labelencoder, onehotencoder, sc] = import_model(bus_line)
    
    cleaned_bus_data = clean_data(bus_data, time_now)
#    print(bus_data)
    encoded_bus_data = encode_data(cleaned_bus_data, labelencoder, onehotencoder, sc)
    
    
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
    
#    output = jsonify(output)
  
    return output








#bus_line = '1'
##usr_lon = 98.3663
##usr_lat = 7.89635
#usr_lon = 98.401550
#usr_lat = 7.862933
#
#data_dict = request_prediction_all_line(usr_lat, usr_lon)
#
#import json
#json.dumps(data_dict)



predict_location('1', "059049183")

#usr_dir = 'out'
#request_prediction(bus_line, usr_lat, usr_lon, usr_dir)
#request_prediction_all_line(usr_lat, usr_lon)


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










