import pandas as pd
import numpy as np

import cleaning_data

SAVED_MODEL_PATH = 'saved-model/'


def get_modellers(bus_line, model_type, load_new_data=False):
    """ Train Keras Deep Learning Model """
    
    def _get_X_y(made_data, model_type):
        """ Get X and y for Deep learning Model """
    
        if model_type == 'time':
            X_cols = ['day_of_week', 'status', 'hour', 'distance_to_next',
                    'speed', 'linear_ref']
            y_col = ['time_to_next']
            
        elif model_type == 'location':
            X_cols = ['day_of_week', 'status', 'hour', 'distance_to_next',
                    'speed', 'linear_ref']
            y_col = ['time_to_next']
        
        X = made_data[X_cols].values
        y = made_data[y_col].values
        
        return X, y
    
    ## Import cleaned data, the result from cleaning_data.py
    if load_new_data:
        made_data, data = cleaning_data.clean_data(bus_line)
    else:
        DATA_PATH = "data/cleaned_potong{}.csv.gz".format(bus_line)
        made_data = pd.read_csv(DATA_PATH)

    
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
    regressor.fit(X_train, y_train, batch_size=32, nb_epoch=150)

    return [regressor, labelencoder, onehotencoder, sc_X, X_test, y_test]

def save_model(modellers, xy, bus_line, model_type):
    """ Save models to disk
        1. regressor - Deep learning structure to model.json & weight to model.h5
        2. encoders to encoders.pkl
        3. X_test, y_test to Xy.pkl for testing the model"""
        
    from sklearn.externals import joblib
    
    [regressor, labelencoder, onehotencoder, sc] = modellers
    [X_test, y_test] = xy
    
    # serialize model to JSON
    model_json = regressor.to_json()
    with open("{}{}/{}/model.json".format(SAVED_MODEL_PATH, model_type, bus_line), "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    regressor.save_weights("{}{}/{}/model.h5".format(SAVED_MODEL_PATH, model_type, bus_line))
    print("Saved model to disk")
    
    joblib.dump([labelencoder, onehotencoder, sc], "{}{}/{}/encoders.pkl".format(SAVED_MODEL_PATH, model_type, bus_line))
    joblib.dump([X_test, y_test], "{}{}/{}/Xy.pkl".format(SAVED_MODEL_PATH, model_type, bus_line))

def run(bus_line, model_type):
    
    [regressor, labelencoder, onehotencoder, sc_X, X_test, y_test] = get_modellers(bus_line, model_type)

    if regressor is None:
        return None
    save_model([regressor, labelencoder, onehotencoder, sc_X], [X_test, y_test], bus_line, model_type)


run('1', 'time')
run('2', 'time')
run('2a', 'time')
run('3', 'time')

#run('1', 'location')
#run('2', 'location')
#run('2a', 'location')
#run('3', 'location')






