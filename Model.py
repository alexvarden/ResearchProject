from matplotlib.backends.backend_pdf import PdfPages
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import FunctionTransformer
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors

class Model:
    def __init__(self, name, 
                 categorical_features=[], 
                 continous_features=[],
                 date_features=[],
                 drop_features=[],
                 classname='class'
                 ):
       
        self.continous_features = continous_features
        self.categorical_features = categorical_features
        self.date_features = date_features
        self.drop_features = drop_features
        self.classname = classname
        self.name = name

    def load_data(self):
        data = pd.read_csv(f'data/{self.name}.csv')
        self.X = data.drop([self.classname], axis=1)
        self.y = data[self.classname]

    def drop_columns(self,x, columns):
        return x.drop(columns=self.drop_features)

    def getPipline(self, model):
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        transformations = ColumnTransformer(transformers=[
            ('cat', categorical_transformer, self.categorical_features),
            ('num_preprocess', MinMaxScaler(), self.continous_features),
            ('drop_columns', FunctionTransformer(self.drop_columns, validate=False), self.drop_features)
        ])

        return Pipeline(steps=[('preprocessor', transformations),
                            ('classifier', model)])
    def train(self):
        self.clf = self.getPipline(self.getModel())
        self.clf.fit(self.X_train, self.y_train)
        self.saveModel()

    def getFilename(self):
        return f"models/{self.name}-{self.modelName}-pipe.pickle"

    def saveModel(self):
        pickle.dump(self.clf, open(self.getFilename(), "wb"))

    def loadModel(self):
        self.clf = pickle.load(open(self.getFilename(), "rb"))

    def getLocalisedData(self, k, example):
        nn = self.getPipline(NearestNeighbors(n_neighbors=k))
        nn.fit(self.X)

        distances, indices = nn['classifier'].kneighbors(self.X)
        example_index = self.X.index.get_loc(example.name)
        k_nearest_neighbors = self.X.iloc[indices[example_index, 1:k+1]]
        k_nearest_neighbors[self.classname] = self.y.iloc[indices[example_index, 1:k+1]]

        return k_nearest_neighbors




