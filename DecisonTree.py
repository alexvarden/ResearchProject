from matplotlib.backends.backend_pdf import PdfPages
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
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
from Model import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import roc_auc_score


class DecisonTree(Model):
    def __init__(self, name,
        categorical_features=[],
        continous_features=[],
        classname='class',
        fileModifer="Tree"
    ):
        super().__init__(name,
            categorical_features=categorical_features,
            continous_features=continous_features,
            classname=classname,

        )
        self.regression = False
        self.modelName = "tree"
        self.maxDepth = None
        self.fileModifer = fileModifer

    def setValidationData(self,data):
        self.X_test = data.drop([self.classname], axis=1)
        self.X_test = self.X_test.loc[:, ~ self.X.columns.str.contains('^Unnamed')]
        self.y_test = data[self.classname]

    def setTrainingData(self, data):
        self.X_train = data.drop([self.classname], axis=1)
        self.X_train = self.X_train.loc[:, ~ self.X.columns.str.contains('^Unnamed')]
        self.y_train = data[self.classname]

    def getModel(self):
        if(self.regression):
            return DecisionTreeRegressor(max_depth=self.maxDepth)
        else:
            return DecisionTreeClassifier(max_depth=self.maxDepth)
            
    def evaluate(self, fileMod="tree"):
        
        # Multiclass classification requires One verses Rest in order to compare ROC_AUC
        self.y_score = self.clf.predict_proba(self.X_test)
        
        # mean_path_length = get_mean_path_length(clf)

        y_pred = self.clf.predict(self.X_test)


        depth = self.clf['classifier'].tree_.max_depth
        if self.regression:
            # # The mean squared error
            print('Mean squared error: %.2f'
                  % mean_squared_error(self.y_test, y_pred))
            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination: %.2f'
                  % r2_score(self.y_test, y_pred))
        else:
            self.roc_curves()

            if (len(self.y_train.unique()) > 2):
                auc_var = self.getMeanAuc()
            else:
                auc_var = roc_auc_score(self.y_test, self.y_score[:, 1])

            print("AUC score:", auc_var)
            print('Accuracy:', metrics.accuracy_score(self.y_test, y_pred))

        # Evaluate the accuracy of the model
        print('Depth of the tree:', depth)

        if self.regression:
            print("unable to save tree for regression")
        else:
            self.saveTreeClassifcationToFile(
                self.clf, fileMod, classnames=self.y_train.unique())

    def getMeanAuc(self):
        # I hate i had to do this myself, but the roc_auc_score() cant handle having a varialbe test set for diffrent clasees
        label_binarizer = LabelBinarizer().fit(self.y_train)
        y_onehot_test = label_binarizer.transform(self.y_test)
        aggragtedAuc = []
        for class_of_interest in label_binarizer.classes_:
            class_id = np.flatnonzero(
                label_binarizer.classes_ == class_of_interest)[0]
            y_onehot_test[:, class_id]
            self.y_score[:, class_id]
            fpr, tpr, _ = roc_curve(
                y_onehot_test[:, class_id], self.y_score[:, class_id],
            )
            roc_auc = auc(fpr, tpr)
            # print(f"{class_of_interest} auc: {roc_auc}")
            aggragtedAuc.append(roc_auc)
        return np.mean(aggragtedAuc)

    def saveTreeClassifcationToFile(self, model, fileMod="tree", classnames=None):
        dot_data = export_graphviz(model['classifier'],
            out_file=None,
            feature_names=model['preprocessor'].get_feature_names_out(
        ),
            class_names=classnames,
            filled=True,
            rounded=True,
            special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(
            f'decision_trees/{fileMod}-{self.fileModifer}-{self.name}')

    def roc_curves(self):
        num_classes = len(np.unique(self.y_train))
        print(np.unique(self.y_train))
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(
                self.y_test, self.y_score[:, 1], pos_label=np.unique(self.y_train)[1])
            roc_auc = auc(fpr, tpr)

            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name=self.name)
            roc_display.plot()
            plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
            plt.legend()
            plt.savefig(f'ROC/{self.name}_nn_roc_curve.pdf')
            plt.close()

        else:
            self.label_binarizer = LabelBinarizer().fit(self.y_train)
            self.y_onehot_test = self.label_binarizer.transform(self.y_test)

            with PdfPages(f'ROC/{self.name}_nn_roc_curve.pdf') as pdf:
                for class_of_interest in self.label_binarizer.classes_:
                    class_id = np.flatnonzero(
                        self.label_binarizer.classes_ == class_of_interest)[0]

                    RocCurveDisplay.from_predictions(
                        self.y_onehot_test[:, class_id],
                        self.y_score[:, class_id],
                        name=f"{class_of_interest} vs the rest",
                        color="darkorange",
                    )
                    plt.plot([0, 1], [0, 1], "k--",
                             label="chance level (AUC = 0.5)")
                    plt.axis("square")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"{class_of_interest} vs the rest")
                    plt.legend()
                    pdf.savefig()
                    plt.close()
