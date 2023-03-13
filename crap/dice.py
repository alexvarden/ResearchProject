from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn import tree
import pickle
import dice_ml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('data/adult.csv')
target = data["class"]

# Split data into train and test
dataX = data.drop("class", axis=1)
x_train, x_test, y_train, y_test = train_test_split(dataX,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=target)

numerical = ["age", "hours-per-week"]
categorical = x_train.columns.difference(numerical)

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier())])
model = clf.fit(x_train, y_train)


d = dice_ml.Data(dataframe=data, continuous_features=[
                 'age', 'hours-per-week'], outcome_name='class')
m = dice_ml.Model(model=model, backend="sklearn")


exp = dice_ml.Dice(d, m, method="genetic")
query_instance = x_train[1:2]
e1 = exp.generate_counterfactuals(query_instance, total_CFs=10, 
                                  desired_class="opposite",
                                   features_to_vary="all")
e1.visualize_as_dataframe(show_only_changes=True)
