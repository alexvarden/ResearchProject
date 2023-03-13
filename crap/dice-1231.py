# %%
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn import tree
from imblearn.pipeline import Pipeline
pd.options.display.max_rows = 4000



# %%
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

# %%
X = data.drop(['class'], axis=1)
y = data['class']
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=1)
# print(X_train.shape, y_train.shape)


# %%
import pickle

filename = "models/adult-nn-pipe.pickle"
clf = pickle.load(open(filename, "rb"))

 

# %%
import dice_ml


# Dataset for training an ML model
d = dice_ml.Data(dataframe=data,
                 categorical_features=cat,
                 continuous_features=continuous_features,
                 outcome_name='class')

# model = Pipeline(steps=[
#     ('preprocess', pipeline),
#     ('MultiLayerPercptron', clf)
# ])

# Pre-trained ML model
m = dice_ml.Model(model=clf, backend="sklearn", model_type='classifier')
# DiCE explanation instance
exp = dice_ml.Dice(d, m, method="genetic",)


# %%
import json
import csv

counterFactuals = []
percentage_to_sample = 0.2
# percentage_to_sample = percentage_to_sample * 0.1

f = open('counterfactuals/adult-1.csv', 'w')
first = True
writer = csv.writer(f)
result = exp.generate_counterfactuals(
    X.sample(frac=percentage_to_sample,random_state=1),
    total_CFs=1, 
    # desired_class='>40k',
    # proximity_weight=0.8,
    # sparsity_weight=0.2, 
    # diversity_weight=0.6, 
    # categorical_penalty=0.1,
    # stopping_threshold=0.4,
    verbose=True
)
result = json.loads(result.to_json())
cfList = result["cfs_list"]
cfList = flattenArray(cfList)
# cfList = lookupClassLabel(cfList, classCodes)
if(first):
    print(result['feature_names_including_target'])
    writer.writerow(result['feature_names_including_target'])
    first = False
writer.writerows(cfList)




# %%


# %% [markdown]
# Validate Cfs against model, should get 100% accruracy as these are informed by model

# %%
from sklearn.preprocessing import LabelBinarizer


cfdata = pd.read_csv('random-pertuabtions/dry-been-100.csv')
# print(cfdata.head())

X = cfdata.drop(['class'], axis=1)
y = cfdata['class']


y_score = clf.predict_proba(X)

label_binarizer = LabelBinarizer().fit(y)
y_onehot_test = label_binarizer.transform(y)
print(y_onehot_test.shape)  # (n_samples, n_classes)

for class_of_interest in label_binarizer.classes_:
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{class_of_interest} vs the rest")
    plt.legend()
    plt.show()


# %%



# %%
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
        elif (data[columnName].dtype.name == 'float64' ):
            value = random.uniform(
                data[columnName].min(), data[columnName].max())
        elif (data[columnName].dtype.name == 'int64'):
            value = random.uniform(
                data[columnName].min(), data[columnName].max())
        rows[columnName].append(value)

randomData = pd.DataFrame(data=rows, columns=data.columns)

X = randomData.drop(['class'], axis=1) 




randomData['class'] = clf.predict(X)

print(randomData.groupby(['class']).size())



randomData.to_csv('random-pertuabtions/dry-been-100.csv',index=False)


# %% [markdown]
# 

# %%
clf.predict(X)


# %%


# %%



