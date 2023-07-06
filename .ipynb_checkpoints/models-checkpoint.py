import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

from utils import load_pickle, dump_pickle
from preprocess import onehotencoding_and_standardize




def NN(n_inputs, n_units=10, dropout=0.1, l1_reg=0.001, activation='relu', L=3):
    # L>0 is the number of hidden layers
    model = Sequential()
    model.add(Dense(units=n_units, input_dim=n_inputs, kernel_regularizer=l1(l1_reg), kernel_initializer='normal', activation=activation))
    model.add(Dropout(dropout))
    for i in range (0, L-1):
        model.add(Dense(units=n_units, kernel_regularizer=l1(l1_reg), kernel_initializer='normal', activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal')) 
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['mae'])
    
    return model


def parameter_tuning(X, y,modeltype, cv=3, n_epochs=100, n_batch=50, seed=1234):

    if modeltype == 'NN':
        param_grid = dict(n_inputs= [X.shape[1]], dropout=[0,0.01,0.1], n_units=[10, 20, 50], l1_reg = [0, 0.0001, 0.001], activation=['relu','tanh']) 
        estimator = KerasRegressor(build_fn=NN, epochs=n_epochs, batch_size=n_batch, verbose=1)   
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, verbose=1)
        grid_result = grid_search.fit(X, y)
    
    if modeltype == 'RF':
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
        
        estimator = RandomForestRegressor()
        grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = cv, n_jobs = -1, verbose = 1)
        grid_result = grid_search.fit(X, y)

    return grid_result.best_params_



def train_models(X_train,y_train,modeltype, tune=False):
    if tune == True:
        if modeltype == 'RF':
            best_params = parameter_tuning(X_train, y_train,modeltype, cv=3, n_epochs=100, n_batch=50, seed=1234)
            n_estimators = best_params['n_estimators']
            max_features = best_params['max_features']
            max_depth = best_params['max_depth']
            min_samples_split = best_params['min_samples_split']
            min_samples_leaf = best_params['min_samples_leaf']
            bootstrap = best_params['bootstrap']
            
            model = RandomForestRegressor(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,
                                          min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,bootstrap=bootstrap)
            model.fit(X_train,y_train,verbose=1)
            dump_pickle(model,'./model/model_tune_%s.p'%(modeltype))
        
        if modeltype == 'NN':
            best_params = parameter_tuning(X_train, y_train,modeltype, cv=3, n_epochs=100, n_batch=50, seed=1234)
            n_inputs = X_train.shape[1]
            L = 3     ###you can tune the number of layers as well if you want
            n_units = best_params['n_units']
            l1_reg = best_params['l1_reg']
            activation = best_params['activation']
            drop_out = best_params['dropout']
            
            model = NN(n_units=n_units, n_inputs=n_inputs, dropout=drop_out, 
                       l1_reg=l1_reg, activation=activation, L=L)
            model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1)
            model.save('./model/model_tune_%s.h5'%(modeltype))
    else:
        if modeltype == 'RF':
            model = RandomForestRegressor(n_estimators=1000,max_features='sqrt')
            model.fit(X_train,y_train)
            dump_pickle(model,'./model/model_%s.p'%(modeltype))
        if modeltype == 'NN':
            n_inputs = X_train.shape[1]
            n_epochs = 100 # maximum number of epochs (to be used with early stopping)
            n_batch = 50 # mini-batch size
            model = NN(n_inputs, n_units=10, dropout=0.1, l1_reg=0.001, activation='relu', L=3)
            model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=1)
            model.save('./model/model_%s.h5'%(modeltype))
            
    return model
            