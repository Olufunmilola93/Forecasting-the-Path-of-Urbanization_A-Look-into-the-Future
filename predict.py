import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, mean_squared_error


###predict and evaluate(if you have the test target values)
def predict_evaluate(X_test,y_test,model):
    y_pred = model.predict(X_test)
    if y_test['target'].isnull().values.any() == False:
        print('RMSLE is:', np.sqrt(mean_squared_log_error(y_test['target'], y_pred )))
    else:
        pass
    return pd.concat([y_test[['tile_h','tile_v','target']].reset_index(drop=True),pd.DataFrame(y_pred,columns=['prediction'])], axis=1)