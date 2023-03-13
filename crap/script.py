from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn import tree
import pickle
import json
import csv


dataset = "dry-bean"
modelType = "nn"

filename = f"models/{dataset}-{modelType}.pickle"
clf = pickle.load(open(filename, "rb"))
continuousFeatures=[
    'Area',
    'Perimeter',
    'MajorAxisLength',
    'MinorAxisLength',
    'AspectRation',
    'Eccentricity',
    'ConvexArea',
    'EquivDiameter',
    'Extent',
    'Solidity',
    'roundness',
    'Compactness',
    'ShapeFactor1',
    'ShapeFactor2',
    'ShapeFactor3',
    'ShapeFactor4'
]

data = pd.read_csv(f'data/{dataset}.csv')
data.head()
data['class'] = pd.Categorical(data['class'])
classCodes = dict(enumerate(data['class'].cat.categories))

X = data.drop(['class'], axis=1)
y = data['class']
import dice_ml

# Dataset for training an ML model
d = dice_ml.Data(dataframe=data,
                continuous_features=continuousFeatures,
                outcome_name='class')

# Pre-trained ML model
m = dice_ml.Model(model=clf, backend="sklearn", model_type='classifier')
# DiCE explanation instance
exp = dice_ml.Dice(d, m, method="genetic",)



counterFactuals = []

f = open('counterfactuals/dry-bean-1.csv', 'w')
first = True
writer = csv.writer(f)
for classCode in classCodes:
    query_instances = data[data["class"].cat.codes ==classCode]
    for desiredClass in classCodes:
        if (desiredClass == classCode):
            continue
        result = exp.generate_counterfactuals(
            query_instances.drop(['class'], axis=1),
            total_CFs=1, 
            desired_class=int(desiredClass),
            proximity_weight=0.2,
            sparsity_weight=0.2, 
            diversity_weight=5.0, 
            categorical_penalty=0.1,
            stopping_threshold=1

        )
        result = json.loads(result.to_json())
        cfList = result["cfs_list"]
        cfList = flattenArray(cfList)
        cfList = lookupClassLabel(cfList, classCodes)

        if(first):
            print(result['feature_names_including_target'])
            writer.writerow(result['feature_names_including_target'])
            first = False
        writer.writerows(cfList)







import random



samples = 1000
rows = {}

headings = list(data.columns.values)



for columnName in headings:
    rows[columnName] = []


for _ in range(samples):


    for columnName in headings:
        # print(columnName, data[columnName].dtype.name)
        if (data[columnName].dtype.name == 'category'):
            value = None
        elif (data[columnName].dtype.name == 'float64'):
            value = random.uniform(
                data[columnName].min(), data[columnName].max())
        rows[columnName].append(value)

randomData = pd.DataFrame(data=rows, columns=data.columns)

X = randomData.drop(['class'], axis=1) 
randomData['class'] = clf.predict(X)

print(randomData.groupby(['class']).size())


randomData.to_csv(f"random-pertuabtions/{dataset}-{modelType}-100.csv", index=False)
