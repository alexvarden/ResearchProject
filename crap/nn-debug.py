from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline


from PreproccessPipeline import *


data = pd.read_csv('data/adult.csv')

cat = ['workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'sex', 'native-country']

continuous_features = [
    'age',
    'fnlwgt',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'education-num'
]

pipeline = PreproccessPipeline(continuous=continuous_features,
                               categories=cat, className="class")

# data = pipeline.fit_transform(data_og)

# print(data.head())

# data = pipeline.labelDecode(data)
data.head()


# %% [markdown]
# 
# Splitting data into training / test allows me to test the accuracy of my model on unseend data.
# 
# this split is random however to mainatin class raitios i have used stratisfied sampling.
# 
# random state = 1 allows randomisation to remain deterministic, this ensures that test data dos not bleed into the training data

from imblearn.over_sampling import SMOTE

X = data.drop(['class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)

print(y_train.value_counts())


# print(data.head())

print(data.info(verbose=1))

model = MLPClassifier(random_state=1,
        max_iter=700,
        activation="relu",
        hidden_layer_sizes=(100, 100, 100, 100),
        verbose=True,
        solver="adam"
)


sm = SMOTE()
# X_train, y_train = sm.fit_resample(X_train, y_train)


clf = Pipeline(steps=[
    ('preprocess', pipeline),
    ('MultiLayerPercptron', model)
])

trainedModel = clf.fit(X_train, y_train)



import pickle

filename = "models/adult-nn-pipe.pickle"
pickle.dump(trainedModel, open(filename, "wb"))


y_score = trainedModel.predict_proba(X_test)

RocCurveDisplay.from_estimator(
    trainedModel,
    X_test, y_test,
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f">50K")
plt.legend()
plt.show()

