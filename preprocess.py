####preprocessing 
import pandas as pd
from make_data import datapipeline
from sklearn.preprocessing import StandardScaler


def onehotencoding_and_standardize(data):
    '''The function (i) onehot encode kmeans (ii) standardize train and test set features'''
    data = pd.get_dummies(data, columns=['kmeans'])
    explanatory_variables = set(data.columns)-{'block_size','tile_h','tile_v','target','has_target'}
    n = len(data[data['has_target']==1])
    
    X_train = data[data['has_target']==1][list(explanatory_variables)][:int(0.8*n)]
    y_train = data[data['has_target']==1]['target'][:int(0.8*n)]
    
    X_test = data[data['has_target']==1][list(explanatory_variables)][int(0.8*n):]
    y_test = data[data['has_target']==1][['tile_h','tile_v','target']][int(0.8*n):]
    
    sc = StandardScaler()
    sc.fit(X_train), sc.fit(X_test)
    
    return sc.transform(X_train), y_train, sc.transform(X_test), y_test  
