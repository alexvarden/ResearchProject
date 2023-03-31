from matplotlib.backends.backend_pdf import PdfPages
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Model import * 

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

class RF_Regressor(Model):
    def __init__(self, name,
                 categorical_features=[],
                 continous_features=[],
                 classname='cnt',
                 max_depth=5,
                 n_estimators=100):

        super().__init__(name,
            categorical_features=categorical_features,
            continous_features=continous_features,
            classname=classname
        )

        self.modelName = "RF"
        print(f"******** {name} {self.modelName} ********")
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2,  random_state=1)

    def getModel(self):
        return RandomForestRegressor(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            random_state=0
        )

    def evaluate(self):

        # Multiclass classification requires One verses Rest in order to compare ROC_AUC
        # self.y_score = self.clf.predict_proba(self.X_test)
        self.clf.out_activation_ = 'logistic'

        y_reg_pred_test = self.clf.predict(self.X_test)

        print('Mean squared error: %.2f'
            % mean_squared_error(self.y_test, y_reg_pred_test))
        print('Coefficient of determination: %.2f'
            % r2_score(self.y_test, y_reg_pred_test))
        print('Mean Absolute Error: %.2f'
            % mean_absolute_error(self.y_test, y_reg_pred_test))
        
        with PdfPages(f'ROC/{self.name}_nn_ressidual_plot.pdf') as pdf:
            plt.scatter(self.y_test, y_reg_pred_test, s=5, color='black')
            plt.plot([0, self.y_test.max()], [0, self.y_test.max()], color='red')
            plt.xlabel("Actual")
            plt.ylabel("Prediction")

            plt.title(f"{self.name}")
            pdf.savefig()
            plt.close()



    