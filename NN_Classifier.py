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


class NN_Classifier(Model):
    def __init__(self, name,
                 categorical_features=[],
                 continous_features=[],
                 date_features=[], 
                 hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100)):
    
        super().__init__(name,
                       categorical_features=categorical_features,
                       continous_features=continous_features,
                       date_features=date_features)

        self.modelName = "nn"
        self.hidden_layer_sizes = hidden_layer_sizes

        print(f"******** {name} {self.modelName} ********")

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=1)


    def getModel(self):
        return MLPClassifier(random_state=1,
            max_iter=700,
            activation="relu",
            hidden_layer_sizes=self.hidden_layer_sizes,
            verbose=True,
            solver="adam"
        )

    def evaluate(self):
        # Multiclass classification requires One verses Rest in order to compare ROC_AUC
        self.y_score = self.clf.predict_proba(self.X_test)
        self.roc_curves()

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
