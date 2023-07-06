import numpy as np
import pandas as pd
import timeit
from preprocess import onehotencoding_and_standardize
from models import train_models
from predict import predict_evaluate
from utils import load_pickle, dump_pickle
from make_data import kmeans,load_data


def pipeline(modeltype,k,points,tune=False,fe=False):
    '''
    specify (i) the modeltype ('RF' or 'NN') (ii) if you want to do some hyperparametr tuning (tune=True)
    (iii) do you want to perform the feature engineering step (it will take time specify fe=True) or simply use what has been done.
    Output: prediction file. (iv) k is the number of clusters determine from the elbow curve (v)points is the specitifed length of bounded box around a tile.
    '''
    if fe == True:
        finaldata = datapipeline(k,points)
    else:
        #finaldata = load_pickle('./data/finaldata.p')
        finaldata = kmeans(load_data(),k)
        
    X_train, y_train, X_test, y_test = onehotencoding_and_standardize(finaldata)
    
    start_time = timeit.default_timer() 
    model = train_models(X_train,y_train,modeltype, tune=tune)
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: " + str(elapsed) + " (s)")
    predict_evaluate(X_test,y_test,model).to_csv('./result/prediction_%s.csv'%(modeltype))
    return predict_evaluate(X_test,y_test,model)
 
