# Spatial-ML-Predicting-Evolution-of-Urbanization

We predict the evolution of urbanization of 1km2 tiles based on geographic features of the tiles.

Example:

from completepipeline import pipeline

modeltype = 'RF'

k = 4

points = 5

tune = False

fe = False

prediction = pipeline(modeltype,k,points,tune=False,fe=False)

## Project Organization

```
    ├── LICENSE
    ├── README.md           <- The top-level README for authors using this project.
    ├── utils.py            <- utilities used in the project.
    ├── graphs.py           <- this module contain scripts for visualization e.g. elbow plot
    ├── make_data.py        <- this module contains submodules that load athe data and perform feature engineering
    ├── preprocess.py       <- this module allows for onehotencoding and standardization
    ├── models.py           <- this module contains submodules for RandomForestRegression and FeedForward Neural network models
                              it also allows for hyper parameter tuning.
    ├── predict.py          <- this module allows for prediction and evaluation(if the target value of the test set exist)
    ├── completepipeline.py <- this module runs the whole script from feature engineering to model 
                               by calling the function pipeline and providing Arg: modeltype= 'RF' or 'NN', k = number of clusters,
                               points = length of bounded box, tune=False (True if you want to perform parameter tuning), 
                               fe=False (True if you want to perform the feature engineering).            <- 
    ├── data
    │   ├── finaldata.p            <- Final data resulting from feature engineering
    │   ├── analytical_table.csv   <- tabular data with predictors
    │   ├── extrafeatures.csv      <- contains the geometry (area and perimeter) extracted from the shapefile
    │   └── target.csv             <- The target variable for each tiles.
    ├── model                     <- where all final models are stored
    ├── notebook                   <- Jupyter notebook. you can import and run the model in the notebook.                
    ├── setup.py                   <- Allows installing this project's python code for Notebook importation
    ├── __init__.py                <- Allows installing this project's python code for Notebook importation
    └── 
```

