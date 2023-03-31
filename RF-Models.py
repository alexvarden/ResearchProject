from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn import tree

from RF_Classifier import *
from RF_Regressor import *
from DecisonTree import *
from CounterfactualSurrogateModel import *
import logging

import time

N_SAMPLES = 200
experimentName = f"global-RF-{N_SAMPLES}"
logging.basicConfig(filename=f'output-{time.time()}.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

datasets = [
    {
        "name":"iris",
        "model": RF_Classifier,
        "continous_features": ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        "category_features":[],
        "regression":False,
        "classname": "class",
        "max_depth":5,
        "n_estimators":100,
    },
    {
        "name":"dry-bean",
        "model": RF_Classifier,
        "continous_features": ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'],
        "category_features": [],
        "regression": False,
        "classname": "class",
        "max_depth":5,
        "n_estimators":100,

    },
    {
        "name":"adult",
        "model": RF_Classifier,
        "continous_features": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num'],
        "category_features": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
        "regression": False,
        "classname": "class",
        "max_depth":5,
        "n_estimators":100,

    },
    {
        "name":"bike-sharing-hourly",
        "model": RF_Regressor,
        "continous_features": ['dteday', 'holiday', 'weekday', 'workingday', 'season', 'yr', 'mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed'],
        "category_features": ['weathersit'],
        "regression": True,
        "classname":"cnt",
        "max_depth":100,
        "n_estimators":500,

    },
    {
        "name":"wine-quality",
        "model": RF_Regressor,
        "continous_features": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "is_red", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        "category_features": [],
        "regression": True,
        "classname": "quality",
        "max_depth":100,
        "n_estimators":500,
    },
]



for dataset in datasets:
    print(f"========= {dataset['name']} ===========")
    logging.info(f"========= {dataset['name']} ===========")
    model = dataset['model'](dataset['name'],
        categorical_features=dataset['category_features'],
        continous_features=dataset['continous_features'],
        classname=dataset['classname'],
        max_depth=dataset['max_depth'],
        n_estimators=dataset['n_estimators'],
    )

    model.load_data()
    model.split_data()
    model.train()
    model.loadModel()
    model.evaluate()

    start_time = time.time()

    print("STARTING GENERATION 1")
    model_data = model.getGlobalRandomSample(n_samples=N_SAMPLES)

    model_surrogateModel = CountefactualSurrogateModel(model.name,
    fileModifer=experimentName,
    categorical_features=model.categorical_features,
    continous_features=model.continous_features,
    className=model.classname,
    regression=dataset['regression'],
    n_samples=1,
    modelType="RF"  
    )
    model_surrogateModel.loadModel(model.clf)
    model_surrogateModel.loadDataSet(model_data)
    model_surrogateModel.generate(scale=1)

    print("STARTING GENERATION 2")
    model_surrogateModel.loadData(
        path=f"counterfactuals/{model_surrogateModel.fileModifer}-{model.name}-1.csv")
    model_surrogateModel.n_samples = 1
    model_surrogateModel.generate(scale=1, generation=2)

    end_time = time.time()
    logging.info(f"Elapsed time: {end_time - start_time} seconds")
    logging.info(f"SampleSize : {N_SAMPLES}")

    # validationData = model.getGlobalRandomSample(n_samples=N_SAMPLES)

    # surrogateModel = DecisonTree(dataset['name'],
    #     categorical_features=dataset['category_features'],
    #     continous_features=dataset['continous_features'],
    #     classname=dataset['classname'],
    # )

    # trainingData = pd.concat(
    #     [
    #         pd.read_csv(f"counterfactuals/{experimentName}-{model.name}-1.csv"),
    #         pd.read_csv(f"counterfactuals/{experimentName}-{model.name}-2.csv")
    #     ]
    # )

    # surrogateModel.load_data(data=trainingData)
    # surrogateModel.setTrainingData(trainingData)
    # surrogateModel.setValidationData(validationData)
    # surrogateModel.train()
    # surrogateModel.evaluate()

    # surrogateModel = CountefactualSurrogateModel(dataset['name'],
    #     categorical_features=model.categorical_features,
    #     continous_features=model.continous_features,
    #     className=dataset['classname'],
    #     fileModifer=experimentName,
    #     regression=dataset['regression'],
    #     n_samples=1
    # )

    # validationData = model.getGlobalRandomSample(n_samples=N_SAMPLES)

    # surrogateModel.loadModel(model.clf)
    # # surrogateModel.loadDataSet(model_data)

    # print("CounterFactual")
    # surrogateModel.generateTree(dataset=[1, 2],
    #     localisedValidationData=validationData
    # )

    # print("Control")
    # surrogateModel.generateTree(
    #     localisedData=model.getGlobalRandomSample(n_samples=N_SAMPLES),
    #     localisedValidationData=validationData
    # )

    # end_time = time.time()
    # print("Elapsed time: ", end_time - start_time, "seconds")
    # print(f"SampleSize : {N_SAMPLES}")


# %% [markdown]
# 


