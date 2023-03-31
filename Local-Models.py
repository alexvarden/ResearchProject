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
from NN_Classifier import *
from NN_Regressor import *

from DecisonTree import *
from CounterfactualSurrogateModel import *
import logging

import time

N_SAMPLES = 5000
SCALE = 0.1
logging.basicConfig(filename=f'output-{time.time()}.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

radii = [1.1,1.2,1.3]

datasets = [
    {
        "name":"iris",
        "model": RF_Classifier,
        "modelType":"RF",
        "continous_features": ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        "category_features":[],
        "regression":False,
        "classname": "class",
    },
    {
        "name":"dry-bean",
        "model": RF_Classifier,
        "modelType":"RF",
        "continous_features": ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'],
        "category_features": [],
        "regression": False,
        "classname": "class",
    },
    {
        "name":"adult",
        "model": RF_Classifier,
        "modelType":"RF",
        "continous_features": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num'],
        "category_features": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
        "regression": False,
        "classname": "class",
    },
    {
        "name":"bike-sharing-hourly",
        "model": RF_Regressor,
        "modelType":"RF",
        "continous_features": ['dteday', 'holiday', 'weekday', 'workingday', 'season', 'yr', 'mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed'],
        "category_features": ['weathersit'],
        "regression": True,
        "classname":"cnt",
    },
    {
        "name":"wine-quality",
        "model": RF_Regressor,
        "modelType":"RF",
        "continous_features": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "is_red", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        "category_features": [],
        "regression": True,
        "classname": "quality",
    },
    {
        "name": "iris",
        "model": NN_Classifier,
        "modelType":"nn",
        "continous_features": ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        "category_features":[],
        "regression":False,
        "classname": "class",
    },
    {
        "name": "dry-bean",
        "model": NN_Classifier,
        "modelType":"nn",
        "continous_features": ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'],
        "category_features": [],
        "regression": False,
        "classname": "class",
    },
    {
        "name": "adult",
        "model": NN_Classifier,
        "modelType":"nn",
        "continous_features": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num'],
        "category_features": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
        "regression": False,
        "classname": "class",
    },
    {
        "name": "bike-sharing-hourly",
        "model": NN_Regressor,
        "modelType":"nn",
        "continous_features": ['dteday', 'holiday', 'weekday', 'workingday', 'season', 'yr', 'mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed'],
        "category_features": ['weathersit'],
        "regression": True,
        "classname":"cnt",
    },
    {
        "name": "wine-quality",
        "model": NN_Regressor,
        "modelType":"nn",
        "continous_features": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "is_red", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        "category_features": [],
        "regression": True,
        "classname": "quality",
    },
]

    
for dataset in datasets:
    for radius in radii:

        experimentName = f"local-R{radius}-{dataset['modelType']}"
    
        print(f"========= {dataset['name']} {experimentName} ===========")
        logging.info(f"========= {dataset['name']} {experimentName} ===========")
        model = dataset['model'](dataset['name'],
            categorical_features=dataset['category_features'],
            continous_features=dataset['continous_features'],
            classname=dataset['classname'],
        )
        model.setLogger(logging)

        model.load_data()
        model.split_data()
        # model.train()
        model.loadModel()
        model.evaluate()

        start_time = time.time()

        model_data = model.getLocalisedData(
            99, n_samples=N_SAMPLES, radius=radius, globalSample=50000)

        logging.info(f"local model classes: {model_data[model.classname].unique()} ")

        CF_model = CountefactualSurrogateModel(model.name,
            fileModifer=experimentName,
            categorical_features=model.categorical_features,
            continous_features=model.continous_features,
            className=model.classname,
            regression=dataset['regression'],
            n_samples=1,
        )
        print("STARTING GENERATION 1")
        logging.info("STARTING GENERATION 1")

        CF_model.loadModel(model.clf)
        CF_model.loadDataSet(model_data)
        CF_model.generate(scale=SCALE)

        print("STARTING GENERATION 2")
        logging.info("STARTING GENERATION 2")
        CF_model.loadData(
            path=f"counterfactuals/{CF_model.fileModifer}-{model.name}-1.csv")
        CF_model.n_samples = 1
        CF_model.generate(scale=1, generation=2)
        end_time = time.time()
        logging.info(f"Elapsed time: {end_time - start_time} seconds")
        logging.info(f"SampleSize : {N_SAMPLES}")

        # Treatment Group
        logging.info("Treatment model ==========")

        surrogateModel = DecisonTree(dataset['name'],
            categorical_features=dataset['category_features'],
            continous_features=dataset['continous_features'],
            classname=dataset['classname'],
        )
        surrogateModel.setLogger(logging)
        trainingData = pd.concat(
            [
                pd.read_csv(f"counterfactuals/{CF_model.fileModifer}-{model.name}-1.csv"),
                pd.read_csv(f"counterfactuals/{CF_model.fileModifer}-{model.name}-2.csv")
            ]
        )
        validationData = model_data = model.getLocalisedData(
            99, n_samples=N_SAMPLES, radius=radius, globalSample=50000).sample(n=len(trainingData))

        # validationData = validationData.groupby(model.classname, group_keys=False).apply(
        #     lambda x: x.sample(frac=0.2))

        logging.info(f" surrogate model size : {len(trainingData)}")
        logging.info(f" validationData size : {len(trainingData)}")

        surrogateModel.load_data(data=trainingData)
        surrogateModel.setTrainingData(trainingData)
        surrogateModel.setValidationData(validationData)
        surrogateModel.train()
        surrogateModel.evaluate(logging)

        # Control ======== 
        logging.info("Control model ==========") 

        surrogateModel = DecisonTree(dataset['name'],
            categorical_features=dataset['category_features'],
            continous_features=dataset['continous_features'],
            classname=dataset['classname'],
        )
        surrogateModel.setLogger(logging)

        surrogateModel.load_data(data=model_data)
        surrogateModel.setTrainingData(trainingData)
        surrogateModel.setValidationData(validationData)
        surrogateModel.train()
        surrogateModel.evaluate(logging)
