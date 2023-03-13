from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class PreproccessPipeline(BaseEstimator, TransformerMixin):
    def __init__(self,categories=None, continuous=None, className=None):
        self.labels = {}
        self.categories = []
        self.continuous = []
        self.balanceClass = False
        self.className = "class"

        if categories is not None:
            self.categories = categories
        if continuous is not None:
            self.continuous = continuous
        if className is not None:
            self.className = className

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("transforming")
        X = self.labelEncode(X)
        # X = self.classEncode(X)
        return X
    
    def labelEncode(self, X):
        for column in self.categories:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            self.labels[column] = le
        return X

    # def labelDecode(self, X):
    #     for column in self.categories:
    #         X[column] = self.labels[column].inverse_transform(X[column])
        
    #     X[self.className] = self.labels[self.className].inverse_transform(X[self.className])
    #     return X

    def classEncode(self, X):
        le = LabelEncoder()
        X[self.className] = le.fit_transform(X[self.className])
        self.labels[self.className] = le
        return X


def flattenArray(arr):
    # Create an empty 2D list to hold the converted values
    result = []

    # Loop through each sub-array in the input array
    for sub_arr in arr:
        # Loop through each element in the sub-array
        for element in sub_arr:
            # Append the element to the result list
            result.append(element)

    return result


def lookupClassLabel(array, labels):
    for i in range(len(array)):
        array[i][-1] = labels[array[i][-1]]
    return array

 
def get_quantile_ranges(df, num_quantiles):
    # Get the boundary ranges for the quantiles
    quantile_labels = range(1, num_quantiles+1)
    quantile_boundaries = pd.qcut(df, q=num_quantiles, labels=quantile_labels, retbins=True)[1]
    
    # Store the boundary ranges in a dictionary
    quantile_ranges = {}
    for i, label in enumerate(quantile_labels):
        lower_bound = quantile_boundaries[i]
        upper_bound = quantile_boundaries[i+1]
        quantile_ranges[label] = [lower_bound, upper_bound]
        
    return quantile_ranges