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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import FunctionTransformer


class CountefactualSurrogateModel:
    def __init__(self, name,
        className="class", 
        fileModifer="global", 
        categorical_features=[], 
        continous_features=[], 
        regression=False, 
        n_samples=None):

        self.name = name
        self.className = className
        self.continous_features = continous_features
        self.categorical_features = categorical_features
        self.regression = regression
        self.fileModifer = fileModifer
        
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

    def generate(self, scale=1, generation=1, localisedData=None):
        if (self.regression):
            self.generateRegression(
                scale=scale, generation=generation, localisedData=localisedData)
        else:
            self.generateClassifcation(
                scale=scale, generation=generation, localisedData=localisedData)

    def generateRegression(self, scale=1, generation=1, localisedData=None):
        self.setup()
        quantileRanges = self.get_quantile_ranges(self.data[self.className], 4)
        self.cycleAllClasses(quantileRanges, 
                             scale=scale,
                             generation=generation,
                             localisedData=localisedData)

    def generateClassifcation(self, scale=1, generation=1, localisedData=None):
        self.setup()
        self.data[self.className] = pd.Categorical(self.data[self.className])
        classees = dict(enumerate(self.data[self.className].cat.categories))
        self.cycleAllClasses(classees, 
                             scale=scale, 
                             generation=generation,
                             localisedData=localisedData)


    def cycleAllClasses(self, classes, scale=1, generation=1, localisedData=None):
        if (localisedData is not None):
            data = localisedData
        else:
            data = self.data
        
        percentage_to_sample = 1
        
        if (len(data)>50):
            percentage_to_sample = self._getPercentageToSample(classes, scale)

        f = open(f'counterfactuals/{self.fileModifer}-{self.name}-{generation}.csv', 'w')
        first = True
        writer = csv.writer(f)

        for classCode in classes:
            query_instances = self._getQueryInstances(data, classCode, classes)
            
            for desiredClass in classes:
                if (desiredClass == classCode):
                    continue

                print(f" {classes[classCode]} => {classes[desiredClass]}({percentage_to_sample})")
                
                sample = query_instances.drop([self.className], axis=1).sample(
                    frac=percentage_to_sample, random_state=1)

                if (len(sample)<1):
                    print("skipping due to no examples")
                    continue

                try:
                    result, heading = self._queryDice(
                        sample, desiredClass, classes)
                except:
                    print("============== RETRY ! ===============")
                    print(sample)
                    print(f" errror RETRYing : {classes[classCode]} => {classes[desiredClass]}")
                    try:
                        result, heading = self._queryDice(
                        sample, desiredClass, classes)
                    except:
                        print("============== ERROR ! ===============")
                        print(sample)
                        print(f" errror skipping : {classes[classCode]} => {classes[desiredClass]}")
                        continue


                if (first):
                    writer.writerow(heading)
                    first = False

                writer.writerows(result)

    def _getQueryInstances(self, data, classCode, classes):
        if self.regression:
            return data[
                (data[self.className] >= classes[classCode][0]) &
                (data[self.className] <= classes[classCode][1])
            ]
        else:
            return data[
                data[self.className] == classes[classCode]
            ]

    def _queryDice(self, data, desiredClass, classes):
        if self.regression:
            result = self.diceMl.generate_counterfactuals(
                data,
                total_CFs=self.n_samples,
                desired_range=classes[desiredClass],
                proximity_weight=5,
                sparsity_weight=0.2,
                diversity_weight=0.1,
                categorical_penalty=0,
            )
        else:
            result = self.diceMl.generate_counterfactuals(
                data,
                total_CFs=self.n_samples,
                desired_class=int(desiredClass),
                proximity_weight=5,
                sparsity_weight=1,
                diversity_weight=0.1,
                categorical_penalty=0,
            )

        result = json.loads(result.to_json())
        cfList = result["cfs_list"]
        cfList = self.flattenArray(cfList)

        if (not self.regression):
            cfList = self.lookupClassLabel(cfList, classes)

        return cfList, result['feature_names_including_target']

    def _getPercentageToSample(self, classes, scale):
        percentage_to_sample = 1
        if (len(classes) > 1):
            percentage_to_sample = percentage_to_sample / (len(classes) - 1)
        return percentage_to_sample * scale

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

    def generateTree(self, dataset=[],  localisedData=None, localisedValidationData=None, fileMod="tree"):
        files = []

        if (localisedData is not None):
            files.append(localisedData)

        for generation in dataset:
            files.append(pd.read_csv(
                f'counterfactuals/{self.fileModifer}-{self.name}-{generation}.csv'))

        countfactuals = pd.concat(files)

        X_train = countfactuals.drop([self.className], axis=1)
        y_train = countfactuals[self.className]
   
        if localisedValidationData is None:
            validation = pd.read_csv(f'data/{self.name}.csv')
        else:
            validation = localisedValidationData

        X_test = validation.drop([self.className], axis=1)
        y_test = self.clf.predict(X_test)
        
        clf = self.getPipline(DecisionTreeClassifier(max_depth=20))
        clf.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = clf.predict(X_test)

        print(np.unique(y_test))
        print(np.unique(y_pred))

        depth = clf['classifier'].tree_.max_depth
        # mean_path_length = get_mean_path_length(clf)

        y_pred_proba = clf.predict_proba(X_test)
        

        if (len(y_train.unique())>2):
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        print("AUC score:", auc)

        # Evaluate the accuracy of the model
        print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
        print('Depth of the tree:', depth)
        self.saveTreeToFile(clf, fileMod, classnames=y_train.unique())

    def saveTreeToFile(self, model, fileMod="tree",classnames=None):

        dot_data = export_graphviz(model['classifier'],
            out_file=None,
            feature_names=model['preprocessor'].get_feature_names_out(),
            class_names=classnames,
            filled=True, 
            rounded=True,
            special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(
            f'decision_trees/{fileMod}-{self.fileModifer}-{self.name}')


      


    def getTransformer(self):
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer(transformers=[
            ('cat', categorical_transformer, self.categorical_features),
            ('num_preprocess', MinMaxScaler(), self.continous_features),
        ])

    def getPipline(self, model):
        transformations = self.getTransformer()
        return Pipeline(steps=[('preprocessor', transformations),
                               ('classifier', model)])
