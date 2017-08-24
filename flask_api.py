#!flask/bin/python
#-*- coding:utf-8 -*-

'''
    How to Run Flask Applications with Nginx Using Gunicorn: http://www.onurguzel.com/how-to-run-flask-applications-with-nginx-using-gunicorn/
    
    How 2 use flask reloader without running it twice: https://stackoverflow.com/questions/25504149/why-does-running-the-flask-dev-server-run-itself-twice
    
'''

from flask import Flask, jsonify, make_response, request, Response, json, abort
from flask_cors import cross_origin, CORS
from gevent.wsgi import WSGIServer
from werkzeug.contrib.fixers import ProxyFix
from tinydb import TinyDB, Query
import datetime
import arrow
import copy
# import simplejson as json

from smartphuket.garbage.truck_tracking import Truck_tracking
from smartphuket.potong.predict import Predictor as Potong_predictor
from bmta.prediction.predict import Arrival_time_predictor as BMTA_predictor

from smartphuket.garbage.etc_tester import ETC_tester
from api_functions import *

#Aiy lib
from smartphuket.potong.Aiy import predict_potong_flask as potong_location
from smartphuket.potong.Aiy import predict_potong_time_flask as potong_time
from smartphuket.garbage.Aiy import cluster_autogen_bin
from smartphuket.garbage.Aiy import garbage_truck_api

PORT = 4444
TESTING = False
# testing env config
import os.path
if os.path.exists('test_env.py'):
    TESTING = True
    from test_env import TEST_PORT
    PORT = TEST_PORT
else:
    pass

POTONG_LOG = 'smartphuket/potong/log/predict_log.json'
ROOT_URL = '/test_api/' if TESTING else '/api/'

app = Flask(__name__)
CORS(app)
"""
tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]
"""
potong_predictor = None
truck_tracking = None
bmta_predictor = None
db = TinyDB(POTONG_LOG)

# collect_db = TinyDB('smartphuket/garbage/log/collected_log.json')

@app.route(ROOT_URL+'smartphuket')
def index():
    return "Hello, Osas!\n"


"""
@app.route(ROOT_URL+'smartphuket/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})
    """

#-------- potong arrival time prediction -------------------------------------
# TODO? create URL to train a model
@app.route(ROOT_URL+'smartphuket/predict', methods=['POST'])
def potong_predict():
    if not request.json:
        abort(400)
    else:
        res = potong_predictor.predict(request.json)
        return jsonify(res)

@app.route(ROOT_URL+'smartphuket/predict2', methods=['POST'])
def predict_with_log():
    if not request.json:
        abort(400)

    res = potong_predictor.predict(request.json)
    # save log
    db.insert({'timestamp': '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), 'req_data': request.json, 'result': res})

    return jsonify(res)

@app.route(ROOT_URL+'smartphuket/predict2/log', methods=['GET'])
def get_predict_log():
    return jsonify(db.all())
    
#---------- FROM Aiy -----------------------------------------------------------------------------------
@app.route(ROOT_URL+'smartphuket/predict_potong_current_location/<bus_line>/<bus_id>', methods=['GET'])
def predict_potong_location(bus_line, bus_id):
    return jsonify(potong_location.predict_location(bus_line, bus_id,'smartphuket/potong/Aiy/pickled-data/'))



########### fixed #######
from potong_prediction import PotongLocation, PotongTime
potong_time = PotongTime(model_path='smartphuket/potong/Aiy/saved-model')

@app.route(ROOT_URL+'smartphuket/predict_potong_arrival_time/<bus_line>/<usr_lat>/<usr_lon>/<usr_dir>', methods=['GET'])
def predict_arrival_time(bus_line, usr_lat, usr_lon, usr_dir):
    res = potong_time.request_time_prediction(bus_line, usr_lat, usr_lon, usr_dir)
    return jsonify(res)
 
@app.route(ROOT_URL+'smartphuket/predict_all_potongs_arrival_time/<usr_lat>/<usr_lon>/', methods=['GET'])
def predict_arrival_time_all_line(usr_lat, usr_lon):
    res = potong_time.request_time_prediction_all_line(usr_lat, usr_lon)
    return jsonify(res)
    
####### fixed #####


@app.route(ROOT_URL+'smartphuket/predict_all_potongs_arrival_time_mock/<usr_lat>/<usr_lon>/', methods=['GET'])
def predict_all_potongs_arrival_time_mock(usr_lat, usr_lon):
    with open('smartphuket/potong/Aiy/mock_all_potong_result.json') as f:
        res = json.load(f)
    return jsonify(res)
    
@app.route(ROOT_URL+'smartphuket/start_end_terminals/', methods=['GET'])
def start_end_terminals():
    return jsonify(potong_time.START_END_TERMINALS)
    
    
@app.route(ROOT_URL+'smartphuket/garbage/get_clustered_auto_bins', methods=['GET'])
def get_clustered_auto_bins():
    return jsonify(cluster_autogen_bin.get_clustered_auto_bins())
    
@app.route(ROOT_URL+'smartphuket/garbage/get_bin_route/<vid>/<y>/<m>/<d>/', methods=['GET'])
def get_bin_route_json(vid, y, m, d):
    return jsonify(garbage_truck_api.get_bin_route(vid, int(y), int(m), int(d)))

@app.route(ROOT_URL+'smartphuket/garbage/get_lastest_bin_route/<vid>/', methods=['GET'])
def get_lastest_bin_route(vid):
    return jsonify(garbage_truck_api.get_lastest_bin_route(vid))
#---------- END FROM Aiy -------------------------------------------------------------------------------


#-------- Smart garbage management --------------------------------------------
# TODO get truck sched (when should depart)
@app.route(ROOT_URL+'smartphuket/garbage/binroute', methods=['GET'])
def get_bin_route():
    if not request.args.get('vid'):
        return "<0x01> Please input the truck vehicle id! <\br>\
                The input fmt is: /binroute?vid=VEHICLE_ID [&regis_task=true/false]\n"
    else:
        vid = request.args.get('vid')
        js_out,job_df = truck_tracking.truck_task.get_best_route(vid)
        
        if request.args.get('regis_task') and request.args.get('regis_task')=='true' and job_df:
            truck_tracking.truck_task.add_task(vid, job_df)

    return Response(js_out, mimetype='application/json')
    
@app.route(ROOT_URL+'smartphuket/garbage/collect', methods=['GET'])
def collect_bin():
    if not request.args.get('vid') and request.args.get('bid'):
        return "<0x01> Please input the truck vehicle id and bin id! <\br>\
                The input fmt is: /collect?vid=VEHICLE_ID&bid=BIN_ID\n"
    else:
        vid = request.args.get('vid')
        bid = request.args.get('bid')
        truck_tracking.truck_task.update_task(vid,'collected_bin', int(bid))

    return vid+' collected bin id '+bid

@app.route(ROOT_URL+'smartphuket/garbage/truck_track', methods=['GET'])
def get_garbage_truck():
    if truck_tracking.truck:
        trucks = copy.deepcopy(truck_tracking.truck)
        for vid in trucks:
            for key in ('gps_timestamp','state_begin','timestamp'):
                if key in trucks[vid]:
                    trucks[vid][key] = str(trucks[vid][key])
        return jsonify(trucks)
        
    else:
        return 'ERROR: No truck_tracking instance!'

@app.route(ROOT_URL+'smartphuket/garbage/truck_track/log', methods=['GET'])
def get_garbage_collecting_log():    
    res = []
    cs = None
    if request.args.get('from'):
        try:
            from_time = arrow.get(request.args.get('from')).to('utc').naive
            if request.args.get('to'):
                to_time = arrow.get(request.args.get('to')).to('utc').naive
        except:
            return "Date time format is invalid! <br>\
                    Please use ISO 8601 format, eg. '2017-05-30T10:59:26+00:00'.<br>\
                    The input fmt is: truck_track/log[?from=TIME1&to=TIME2]<br>\
                    Your from arg is "+request.args.get('from')
                
        if request.args.get('to'):  
            if request.args.get('vid'): #also give vehicle id
                cs = truck_tracking.clt_bin.find({
                    "timestamp": {
                        "$gt": from_time,
                        "$lte": to_time
                    },
                    "vid":request.args.get('vid')
                })             
            else:
                cs = truck_tracking.clt_bin.find({
                    "timestamp": {
                    "$gt": from_time,
                    "$lte": to_time
                }}) 
        else: #from w/o to
            if request.args.get('vid'): #also give vehicle id
                cs = truck_tracking.clt_bin.find({
                    "timestamp": {
                        "$gt": from_time
                    },
                    "vid":request.args.get('vid')
                })             
            else: 
                cs = truck_tracking.clt_bin.find({
                    "timestamp": {
                    "$gt": from_time
                }}) 
    elif request.args.get('vid'):
        cs = truck_tracking.clt_bin.find({"vid":request.args.get('vid')})  
    else:
        c = truck_tracking.clt_bin.find({}) # get all
       
    if cs:
        for csi in cs:
            csi['_id'] = str(csi['_id'])
            csi['timestamp'] = str(arrow.get(csi['timestamp']))    
            res.append(csi) 
            
        return jsonify(res) #collect_db.all())
    else:
        return jsonify({'error':'not found!'})


@app.route(ROOT_URL+'smartphuket/garbage/truck_track/add_state_answer', methods=['POST'])
def add_state_answer():
    """ add an answer of state to Mongo
    """
    if not request.json:
        abort(400)
    try:        
        return truck_tracking.add_new_actual_state(request.json)
    except:
        return 'Failed!'
    
       
@app.route(ROOT_URL+'smartphuket/garbage/truck_track/state_log', methods=['GET'])
def get_state_log():    
    res = []
    cs = None
    if request.args.get('from') and request.args.get('to'):
        try:
            from_time = arrow.get(request.args.get('from')).to('utc').naive
            to_time = arrow.get(request.args.get('to')).to('utc').naive
        except:
            return "Date time format is invalid! <br>\
                    Please use ISO 8601 format, eg. '2017-05-30T10:59:26+00:00'.<br>\
                    The input fmt is: truck_track/state_log?from=TIME1&to=TIME2[&vid=VEHICLE_ID]<br>\
                    Your from arg is "+request.args.get('from')
                  
        if request.args.get('vid'): #also give vehicle id
            cs = truck_tracking.clt_state.find({
                "state_begin": {
                    "$gt": from_time,
                    "$lte": to_time
                },
                "vid":request.args.get('vid')
            }, sort=[('state_begin',-1)])             
        else:
            cs = truck_tracking.clt_state.find({
                "state_begin": {
                "$gt": from_time,
                "$lte": to_time
            }}, sort=[('state_begin',-1)])         
        
        for csi in cs:
            csi['_id'] = str(csi['_id'])
            csi['state_begin'] = str(arrow.get(csi['state_begin']))    
            res.append(csi) 
            
        return jsonify(res)
        
    else: 
        return "Please input a range of time to query! <br>\
                The input fmt is: truck_track/state_log?from=TIME1&to=TIME2[&vid=VEHICLE_ID]"
  
  
#TODO       
@app.route(ROOT_URL+'smartphuket/garbage/truck_track/num_truck_in_state', methods=['GET'])
def get_num_truck_in_state_log():  
    url_fmt = 'truck_track/num_truck_in_state?[date=ISO-DATE] or [from=TIME1&to=TIME2]'
    res = []
    cs = None
    if request.args.get('date') or (request.args.get('from') and request.args.get('to')):
        if request.args.get('date'):
            try:
                arrow.get(request.args.get('date'))
            except:
                return return_time_fmt_error(request, url_fmt, 'date', is_time=False)
            res = get_num_truck_in_state(truck_tracking.clt_num_truck_in_state, request.args.get('date'))
            
        else:
            try:
                from_time = arrow.get(request.args.get('from')).to('utc').naive
                to_time = arrow.get(request.args.get('to')).to('utc').naive
            except:
                return return_time_fmt_error(request, url_fmt, arg='from') 
            res = get_num_truck_in_state(truck_tracking.clt_num_truck_in_state, None, 
                    request.args.get('from'), request.args.get('to')) 
                    
        return jsonify(res)
        
    else: 
        return "Please input a date or range of time to query! <br>\
                The input fmt is: "+url_fmt
 

@app.route(ROOT_URL+'smartphuket/garbage/truck_track/add_maintain_truck', methods=['GET'])
def add_maintain_truck():
    ''' Add a truck to BEING_MAINTAINED state
        if not specific start time it will be the current time
    '''
    if request.args.get('vid'):
        if request.args.get('timestamp'):
            try:
                from_time = arrow.get(request.args.get('timestamp')).to('utc').naive
            except:
                return "Date time format is invalid! <br>\
                        Please use ISO 8601 format, eg. '2017-05-30T10:59:26+00:00' or '1498113678'.<br>\
                        The input fmt is: truck_track/add_maintain_truck?vid=VEHICLE_ID[&timestamp=START_TIME]<br>\
                        Your timestamp arg is "+request.args.get('timestamp')            
        else:
            from_time = arrow.utcnow().naive        
        _id = truck_tracking.clt_maintain.insert_one({'vid':request.args.get('vid'), 'timestamp':from_time}).inserted_id
        return str(_id)
    else:
        return "Please input a vid to be added! <br>\
                The input fmt is: truck_track/add_maintain_truck?vid=VEHICLE_ID[&timestamp=START_TIME]"
        

@app.route(ROOT_URL+'smartphuket/garbage/truck_track/remove_maintain_truck', methods=['GET'])
def remove_maintain_truck():
    ''' Remove a truck in BEING_MAINTAINED state
    '''
    if request.args.get('vid'):        
        del_cnt = truck_tracking.clt_maintain.delete_one({'vid':request.args.get('vid')}).deleted_count
        if del_cnt > 0:
            return 'Deleted '+vid+' from maintained trucks'
        else:
            return 'There is no '+vid+' to be removed!'
    else:
        return "Please input a vid to remove! <br>\
                The input fmt is: truck_track/remove_maintain_truck?vid=VEHICLE_ID"
    
    
@app.route(ROOT_URL+'smartphuket/garbage/auto_bin', methods=['GET'])
def get_auto_bin():
    return jsonify(a_tester.get_auto_bin())
    
    
@app.route(ROOT_URL+'smartphuket/garbage/answer_auto_bin', methods=['GET'])
def ans_auto_bin():
    if request.args.get('bid') and request.args.get('is_correct') :  
        bid = request.args.get('bid')
        is_correct = request.args.get('is_correct')
        return a_tester.add_autobin_ans(bid, is_correct=='1')
    else:
        return "Please input a vid to remove! <br>\
                The input fmt is: truck_track/remove_maintain_truck?vid=VEHICLE_ID"
 
 
@app.route(ROOT_URL+'smartphuket/garbage/get_auto_bin_ans', methods=['GET'])
def get_auto_bin_ans():
    return jsonify(a_tester.get_autobin_ans()) 
    
    
# @app.route(ROOT_URL+'smartphuket/garbage/test/<string:latlon>', methods=['GET'])
# def test(latlon):
    # if len(latlon) == 0:
        # return "Please input the current location! <\br>\
                # The input fmt is: /test/LATITUDE,LONGITUDE\n"
    # elif latlon.count(',') != 2:
        # return "Format is not correct! <\br>\
                # The correct input fmt is: /test/LATITUDE,LONGITUDE\n"

    # locs = latlon.split(':')
    # cur_loc = {'lat':float(locs[0].split(',')[0]), 'lon':float(locs[0].split(',')[1])}
    # dst_loc = {'lat':float(locs[1].split(',')[0]), 'lon':float(locs[1].split(',')[1])}

    # return jsonify(test_url())


#-------- bmta arrival time prediction -----------------------------------------
@app.route(ROOT_URL+'bmta/predict', methods=['POST'])
def bmta_predict():
    if not request.json:
        abort(400)
    else:
        res = bmta_predictor.predict(request.json)
        return jsonify(res)
        
        
@app.route(ROOT_URL+'try_raise', methods=['GET'])
def try_raise():    
    raise Exception('raise')
    return 'LOL'


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


# app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
    # import os
    # if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        # print '################### Restarting @ {} ###################'.format(arrow.now())
        # print 'start potong predictor...'
        # potong_predictor = Potong_predictor(model_dir='smartphuket/potong/model')
        # print 'start garbage truck tracking...'
        # truck_tracking = Truck_tracking(log_dir='smartphuket/garbage/log')
        # truck_tracking.start()
        # print 'start bmta predictor...'
        # bmta_predictor = BMTA_predictor(model_dir='bmta/prediction/model')
        
        # a_tester = ETC_tester()        
    # app.run(debug=True, port=4444, use_reloader=True)
        
    print 'start potong predictor...'
    potong_predictor = Potong_predictor(model_dir='smartphuket/potong/model')
    if not TESTING:
        print 'start garbage truck tracking...'
        truck_tracking = Truck_tracking(log_dir='smartphuket/garbage/log')
        truck_tracking.start()
    print 'start bmta predictor...'
    bmta_predictor = BMTA_predictor(model_dir='bmta/prediction/model')
    
    a_tester = ETC_tester()
       
    print 'run flask... @{} - @port {}'.format(arrow.now(), PORT)
    http_server = WSGIServer(('', PORT), app, log=None, error_log='error_log.log')
    http_server.serve_forever()
    
    