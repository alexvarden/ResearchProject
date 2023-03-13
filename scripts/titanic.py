import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, chi2

from sklearn_pandas import DataFrameMapper
import pandas as pd


X, y = fetch_openml(
    "titanic", version=1, as_frame=True, return_X_y=True, parser="pandas"
)

# Alternatively X and y can be obtained directly from the frame attribute:
# X = titanic.frame.drop('survived', axis=1)
# y = titanic.frame['survived']


 
# numeric_features = ["age", "fare"]
# numeric_transformer = Pipeline(
#     steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
# )
categorical_features = ["embarked", "sex", "pclass"]

print(pd.get_dummies(X, columns=categorical_features).info())


categorical_transformer = DataFrameMapper([
    ("embarked", OneHotEncoder()),
    ("sex", OneHotEncoder()),
    ("pclass", OneHotEncoder())

], df_out=True)


print(categorical_transformer.fit_transform(X).info())


# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#         ("cat", categorical_transformer, categorical_features),
#     ]
# )



# clf = Pipeline(
#     steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
# )

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # clf.fit(X_train, y_train)
# # print("model score: %.3f" % clf.score(X_test, y_test))
