import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import load_pickle
import shap
from preprocess import onehotencoding_and_standardize
from tensorflow import keras
from tqdm import tqdm

def number_of_clusters(data,N):
    '''
    N: maximum number of clusters
    This function uses the elbow method to determine the maximum number 
    of clusters by plotting distortions against range of number of clusters.
    '''
    
    cols = list(set(data.columns)-set(['block_size','target','has_target','tile_v','tile_h']))
    distortions = []
    for i in tqdm(range(1,N)):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(data[cols])
        distortions.append(km.inertia_)
    plt.plot(range(1,N), distortions, marker='o')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('./figures/kmeans_elbowplot.png',bbox_inches='tight') 
    plt.show()

    

def shapley(data,modeltype):
    data = pd.get_dummies(data, columns=['kmeans'])
    explanatory_variables = set(data.columns)-{'block_size','tile_h','tile_v','target','has_target'}
    if modeltype == 'RF':
        model = load_pickle('./model/model_RF.p')        
        X_train = onehotencoding_and_standardize(data)[0]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
    if modeltype == 'NN':
        model = keras.models.load_model('./model/model_NN.h5')
        X_train, X_test = onehotencoding_and_standardize(data)[0], onehotencoding_and_standardize(data)[2]
        explainer = shap.DeepExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
    
    return shap.summary_plot(shap_values, X_train, list(explanatory_variables))
    

    
