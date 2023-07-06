import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def dump_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
