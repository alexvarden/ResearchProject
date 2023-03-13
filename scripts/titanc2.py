import csv
import json
import dice_ml
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn import tree
from imblearn.pipeline import Pipeline
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
 
df = pd.read_csv('./scripts/titanic-data.csv')  # reading data
# dropping columns that are not useful for classifcation
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df = df.dropna(axis=0) #dropping nan rows

le = preprocessing.LabelEncoder() #encoding the categorical variables into numericals
df['Sex'] = le.fit_transform(df['Sex']) #{'female': 0, 'male': 1}
df['Embarked'] = le.fit_transform(df['Embarked']) #{'C': 0, 'Q': 1, 'S': 2}

X = df.iloc[:, 1:7] #training features
y = df.iloc[:, 0] #label

train_dataset, test_dataset, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify = y) #train test split
X_train = train_dataset.drop('Survived', axis=1)
X_test = test_dataset.drop('Survived', axis=1)

#model training

model = MLPClassifier(random_state=1,
                    max_iter=700,
                    activation="relu",
                    hidden_layer_sizes=(100, 100, 100, 100),
                    verbose=True,
                    solver="adam"
                    )

model.fit(X_train, y_train)


train_dataset.info(verbose=1)
d = dice_ml.Data(dataframe=train_dataset, continuous_features=['Age', 'Fare'], 
                 outcome_name='Survived')

m = dice_ml.Model(model=model, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")

e = exp.generate_counterfactuals(X_test[0:1], total_CFs=5, desired_class="opposite")
e.visualize_as_dataframe(show_only_changes=True)




# e = exp.generate_counterfactuals(X_test[0:1], total_CFs=5, desired_class="opposite", 
#                                  features_to_vary=['Age'])
# e.visualize_as_dataframe(show_only_changes=True)


# e = exp.generate_counterfactuals(X_test[0:1], total_CFs=5, desired_class="opposite", 
#                                  permitted_range={'Fare': [10,50]})
# e.visualize_as_dataframe(show_only_changes=True)