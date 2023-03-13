import random
from sklearn.preprocessing import LabelBinarizer
import csv
import json
import dice_ml
import pickle
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
import dice_ml
import math

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz
from sklearn.neighbors import NearestNeighbors


class CountefactualSurrogateModel:
    def __init__(self, name, className="class", categorical_features=[], continous_features=[], regression=False, n_samples=None):
        self.name = name
        self.className = className
        self.continous_features = continous_features
        self.categorical_features = categorical_features
        self.regression = regression
        
        if (n_samples is None):
            n_samples = 1
        
        self.n_samples = n_samples
        self.loadData()

    def loadDataSet(self, data):
        self.data = data
        self.X = self.data.drop([self.className], axis=1)
        self.y = self.data[self.className]

    def loadData(self, path=None):
        if (path is None):
            path = f'data/{self.name}.csv'
        
        self.loadDataSet(pd.read_csv(path))

    def loadModel(self, model = None):
        if (model is not None):
            self.clf = model
            return
        
        filename = f"models/{self.name}-nn.pickle"
        self.clf = pickle.load(open(filename, "rb"))

    def setup(self):
        d = dice_ml.Data(
            dataframe=self.data,
            continuous_features=self.continous_features,
            category_features=self.categorical_features,
            outcome_name=self.className
        )
           
        m = dice_ml.Model(
            model=self.clf, 
            backend="sklearn", 
            model_type="regressor" if self.regression else "classifier"
        )
        self.diceMl = dice_ml.Dice(d, m, method="genetic")

    def generate(self, scale=1, generation=1, classes=[]):
        self.setup()
        if (self.regression):
            self.generateRegression(scale=scale,generation=generation)
        else:
            self.generateClassifcation(scale=scale,generation=generation,classes=classes)

    def generateRegression(self, scale=1, generation=1):
        quantileRanges = self.get_quantile_ranges(self.data[self.className], 4)
        self.cycleAllClasses(quantileRanges, 
                             scale=scale,
                             generation=generation)

    def generateClassifcation(self, scale=1, generation=1):
        self.data[self.className] = pd.Categorical(self.data[self.className])
        classees = dict(enumerate(self.data[self.className].cat.categories))
        self.cycleAllClasses(classees, 
                             scale=scale, 
                             generation=generation)

    def cycleAllClasses(self, classes, scale=1, generation=1):
        percentage_to_sample = 1
        if (len(classes)>1):
            percentage_to_sample = percentage_to_sample / (len(classes) - 1)
        percentage_to_sample = percentage_to_sample * scale

        f = open(f'counterfactuals/{self.name}-{generation}.csv', 'w')
        first = True
        writer = csv.writer(f)
        for classCode in classes:
            if self.regression : 
                query_instances = self.data[
                    (self.data[self.className] >= classes[classCode][0]) & 
                    (self.data[self.className] <= classes[classCode][1])
                ]
            else:
                query_instances = self.data[
                    self.data[self.className].cat.codes == classCode
                ]

            for desiredClass in classes:
                if (desiredClass == classCode):
                    continue
                print(
                    f" {classes[classCode]} => {classes[desiredClass]}({percentage_to_sample})")

                instances = query_instances.drop([self.className], axis=1).sample(
                    frac=percentage_to_sample, random_state=1)
                
                if self.regression:
                    result = self.diceMl.generate_counterfactuals(
                        instances,
                        total_CFs=self.n_samples,
                        desired_range=classes[desiredClass],
                        proximity_weight=5,
                        sparsity_weight=0.2,
                        diversity_weight=0.5,
                        categorical_penalty=0.1,
                    )
                else:
                    result = self.diceMl.generate_counterfactuals(
                        instances,
                        total_CFs=self.n_samples,
                        desired_class=int(desiredClass),
                        proximity_weight=5,
                        sparsity_weight=0.2,
                        diversity_weight=0.5,
                        categorical_penalty=0.1,
                    )

                result = json.loads(result.to_json())
                cfList = result["cfs_list"]
                cfList = self.flattenArray(cfList)

                if (not self.regression):
                    cfList = self.lookupClassLabel(cfList, classes)

                if (first):
                    print(result['feature_names_including_target'])
                    writer.writerow(result['feature_names_including_target'])
                    first = False
                writer.writerows(cfList)

    def flattenArray(self,arr):
        # Create an empty 2D list to hold the converted values
        result = []
        # Loop through each sub-array in the input array
        for sub_arr in arr:
            if (sub_arr is None):
                continue
            # Loop through each element in the sub-array
            for element in sub_arr:
                # Append the element to the result list
                result.append(element)

        return result

    def lookupClassLabel(self, array, labels):
        for i in range(len(array)):
            array[i][-1] = labels[array[i][-1]]
        return array

    def get_quantile_ranges(self, df, num_quantiles):
        # Add random noise to the input dataframe to ensure unique quantile boundaries
        df_with_noise = df + np.random.normal(scale=1e-10, size=df.shape)

        # Get the boundary ranges for the quantiles
        quantile_labels = range(1, num_quantiles+1)
        quantile_boundaries = pd.qcut(
            df_with_noise, q=num_quantiles, labels=quantile_labels, retbins=True)[1]

        # Store the boundary ranges in a dictionary
        quantile_ranges = {}
        for i, label in enumerate(quantile_labels):
            lower_bound = math.floor(quantile_boundaries[i])
            upper_bound = math.ceil(quantile_boundaries[i+1]-0.1)
            quantile_ranges[label] = [lower_bound, upper_bound]

        return quantile_ranges

    def generateTree(self,dataset=[1]):
        files = []
        for generation in dataset:
            files.append(pd.read_csv(f'counterfactuals/{self.name}-{generation}.csv'))

        countfactuals = pd.concat(files)

        X_train = countfactuals.drop([self.className], axis=1)
        y_train = countfactuals[self.className]

        # actual = pd.read_csv(f'data/{self.name}.csv').sample(frac=0.5)





        X_test = self.data.drop([self.className], axis=1)
        y_test = self.clf.predict(X_test)
        
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = clf.predict(X_test)
        depth = clf.tree_.max_depth
        # mean_path_length = get_mean_path_length(clf)

        # Evaluate the accuracy of the model
        print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
        print('Depth of the tree:', depth)
        self.saveTreeToFile(clf)

    def saveTreeToFile(self,model):

        dot_data = export_graphviz(model, out_file=None,
                                feature_names=self.X.columns,
                                class_names=self.y.unique(),
                                filled=True, 
                                rounded=True,
                                special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(f'decision_trees/{self.name}')



      

    # def localiseTo(self, k, example):
    #     X = self.data.values
    #     nn = NearestNeighbors(n_neighbors=k)
    #     nn.fit(X)
    #     distances, indices = nn.kneighbors(X)
    #     example_index = self.data.index.get_loc(example.name)
    #     k_nearest_neighbors = self.data.iloc[indices[example_index, 1:k+1]]
    #     return k_nearest_neighbors