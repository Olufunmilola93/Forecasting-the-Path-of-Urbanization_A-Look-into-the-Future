import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils import load_pickle, dump_pickle


def load_data():
    data = pd.read_csv('./data/analytical_table.csv')
    target = pd.read_csv('./data/target.csv')
    extrafeatures = pd.read_csv('./data/extrafeatures.csv')
    return data.merge(target, on =['tile_h','tile_v'], how='left').merge(extrafeatures, on=['tile_h','tile_v'], how='left')


def kmeans(data,k):
    cols = list(set(data.columns)-set(['block_size','target','has_target','tile_v','tile_h']))
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
    return pd.concat([data,pd.DataFrame(km.fit_predict(data[cols]), columns=['kmeans'])], axis=1)


def surrounding_tiles_features(x,points,cols,featuremapp):
    '''
    For each tile, compute the average features of tiles that are within 5km^2 bounded box
    points: 5km^2
    featuremapp: features for each tile.
    '''
    vals = defaultdict(list)
    for i in range(int(x['tile_h']-points),int(x['tile_h']+points+1)):
        for j in range(int(x['tile_v']-points),int(x['tile_v']+points+1)):
            if (i,j) in featuremapp['kmeans'].keys():
                for k in cols:
                    vals[k].append(featuremapp[k][i,j])
    for k in cols:
        x['sur_'+k] = round(np.mean(vals[k]),4)
    return x


def datapipeline(k,points):
    '''
    Provides the final dataset by combining the fucntions: load_data,kmeans,surrounding_tiles_features
    Arg: 
    (i) k  is the number of clusters. This is determined using the elbow plot (check graph folder).
    (ii) points is the 5km2 bounded box around a tile.
    '''
    newdata = kmeans(load_data(),k)
    featuremapp = newdata.set_index(['tile_h','tile_v']).to_dict()
    cols = list(set(newdata.columns)-set(['block_size','target','has_target','tile_v','tile_h']))
    tqdm.pandas()
    newdata = newdata.progress_apply(lambda x: surrounding_tiles_features(x,points,cols,featuremapp),axis=1)
    dump_pickle(newdata,'./data/finaldata.p')
    return newdata