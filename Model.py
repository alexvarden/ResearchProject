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
from sklearn.neighbors import NearestNeighbors, VALID_METRICS_SPARSE

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

    def load_data_from(self,paths):
        data=[]
        for path in paths:
            data.append(pd.read_csv(path))
        data = pd.concat(data)
        self.load_data(data=data)
        
    def load_data(self,data=None):
        if (data is None):
            self.load_data_from([f'data/{self.name}.csv'])
            return
        self.X = data.drop([self.classname], axis=1)
        self.y = data[self.classname]
        self.X = self.X.loc[:, ~self.X.columns.str.contains('^Unnamed')]

    def drop_columns(self,x, columns):
        return x.drop(columns=self.drop_features)

    def getTransformer(self):
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer(transformers=[
            ('num_preprocess', MinMaxScaler(), self.continous_features),
            ('cat', categorical_transformer, self.categorical_features),
            # ('drop_columns', FunctionTransformer(self.drop_columns, validate=False), self.drop_features)
        ], verbose=False, remainder='passthrough')

    def getPipline(self, model):
        transformations = self.getTransformer()
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

    def getLocalisedData(self, index, n_samples=1000, radius=1.5, globalSample=50000):
        example = self.X.iloc[[index]]
        samples = self.getRandomSamples(self.X, n_samples=50000)
        local = self.getNearestNeighbour(samples, example, radius=radius)

        if (len(local) == 0):
            raise Exception(
                'could not find any nearest neighbors, make radius bigger')

        local = self.getRandomSamples(local, n_samples)

        local[self.classname] = self.clf.predict(local)
        return local

    def getGlobalRandomSample(self,  n_samples=1 ):
        data = self.getRandomSamples(self.X, n_samples=n_samples)
        data[self.classname] = self.clf.predict(data)

        return data

    def getRandomSamples(self, data, n_samples=1):
        # Generate a new DataFrame with random values for numerical columns
        randomized_num_df = pd.DataFrame()
        for col in self.continous_features:
            mean = data[col].mean()
            std = data[col].std()

            if (data[data[col] < 0].empty):
                randomized_num_df[col] = np.abs(np.random.normal(
                    mean, std, n_samples))
            else:
                randomized_num_df[col] = np.random.normal(
                    mean, std, n_samples)

        # Generate a new DataFrame with random values for categorical columns
        randomized_cat_df = pd.DataFrame()
        for col in self.categorical_features:
            col_counts = data[col].value_counts()
            col_values = col_counts.index.tolist()
            col_weights = col_counts.values / data.shape[0]
            randomized_cat_df[col] = np.random.choice(
                col_values, size=n_samples, p=col_weights)

        # Combine the numerical and categorical DataFrames into one DataFrame with the same index as the original DataFrame
        randomized_df = pd.concat([randomized_num_df, randomized_cat_df], axis=1)

        return randomized_df

    def getNearestNeighbour(self, data, example, radius=0.2, hamming=False):
        transformations = self.getTransformer()
        transformations.fit_transform(self.X)

        samples = transformations.transform(data)
        example = transformations.transform(example)

        if (hamming):
            nn = NearestNeighbors(metric=self.hamming_distance)
        else:
            nn = NearestNeighbors(metric='minkowski')
       
        nn = nn.fit(samples)
        distance, indices = nn.radius_neighbors(
            example, radius, sort_results=True, return_distance=True,   
        )
        return data[data.index.isin(indices[0])]

    def hamming_distance(self,x, y):
        return np.sum(x != y)

    def plot_mean_and_dist(self, data1, data2):
        # Separate numerical and categorical columns
        num_cols = data1.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = data1.select_dtypes(
            include=['object', 'category']).columns.tolist()

        # Plot mean and distribution of numerical columns before randomization
        fig, axs = plt.subplots(len(num_cols), 2, figsize=(8, 4 * len(num_cols)))
        fig.suptitle('Mean and Distribution before and after randomization')

        for i, col in enumerate(num_cols):
            axs[i, 0].set_title(f"{col}: Before randomization")
            axs[i, 0].axvline(x=data1[col].mean(), color='r', label='Mean')
            axs[i, 0].hist(data1[col], bins=20, alpha=0.5, label='Distribution')
            axs[i, 0].legend()

            # Randomize DataFrame and plot mean and distribution of numerical column
            axs[i, 1].set_title(f"{col}: After randomization")
            axs[i, 1].axvline(x=data2[col].mean(),
                            color='r', label='Mean')
            axs[i, 1].hist(data2[col], bins=20,
                        alpha=0.5, label='Distribution')
            axs[i, 1].legend()

        plt.show()
