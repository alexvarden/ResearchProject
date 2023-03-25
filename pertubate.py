

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import warnings
from NN_Classifier import *
from NN_Regressor import *
from CounterfactualSurrogateModel import *


warnings.filterwarnings(
    "ignore", message="DataFrame is highly fragmented")



# # # # #  # -------------------------------
dryBean_category_features = []
dryBean_continous_features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea',
                              'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

dryBean = NN_Classifier('dry-bean',
    hidden_layer_sizes=(
        16, 50, 50, 50, 50, 50, 50, 50, 50, 1000),
    categorical_features=dryBean_category_features,
    continous_features=dryBean_continous_features,
   
)

dryBean.load_data()
dryBean.split_data()
# dryBean.train()
dryBean.loadModel()
dryBean.evaluate()

example = dryBean.X.iloc[[66]]
# localisedData = dryBean.getLocalisedData(example, 0.2)
# localisedClasses = dryBean.getLocalisedData(example, 0.3)

print(dryBean.X.shape)

# samples = dryBean.getGlobalRandomSample(10)

example = dryBean.X.iloc[[66]]


# # example = s.transform(example)

# # print(dryBean.clf['preprocessor'].transformers[1])


# scaler = MinMaxScaler()
# scaler.fit(dryBean.X[dryBean_continous_features])
# # print(example)
# example[dryBean_continous_features] = scaler.transform(
#     example[dryBean_continous_features])

# examples = generate_perturbations(example,   1000, 0.5)


# examples[dryBean_continous_features] = scaler.inverse_transform(
#     examples[dryBean_continous_features])


# print(examples)



# # #  # -------------------------------
category_features = ['workclass', 'education', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'native-country']

continous_features = ['age', 'fnlwgt', 'capital-gain',
                      'capital-loss', 'hours-per-week', 'education-num']
date_features = []



adult = NN_Classifier('adult',
    hidden_layer_sizes=(100, 100, 100, 100),
    categorical_features=category_features, continous_features=continous_features
)
adult.load_data()
adult.split_data()
# adult.train()
adult.loadModel()
adult.evaluate()
print(adult.X.shape)


def getLocalisedData(self, index , n_samples=1000, radius=1.5, globalSample=50000):
    samples = self.getGlobalRandomSample(adult.X, globalSample)
    example = self.X.iloc[[index]]
    local = self.getLocalisedData(samples, example, radius=radius)
    return self.getGlobalRandomSample(local, n_samples)



