from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import pickle

# Load the Adult dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = pd.read_csv(url, header=None)

# Add column headers
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
              'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
              'hours-per-week', 'native-country', 'income']

# Preprocess the data
# Remove missing values and convert categorical variables to numerical using one-hot encoding
print("Preprocessing the data...")
df = df.dropna()
df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status',
                                 'occupation', 'relationship', 'race', 'sex', 'native-country'])

# Normalize the numerical features
print("Normalizing the numerical features...")
scaler = StandardScaler()
numerical_cols = ['age', 'fnlwgt', 'education-num',
                  'capital-gain', 'capital-loss', 'hours-per-week']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(df.drop('income', axis=1),
                                                    df['income'],
                                                    test_size=0.2,
                                                    random_state=42)

# Use SMOTE to oversample the minority class within the cross validation
# This way the sampling is done on the training data only, hence avoiding leakage
print("Balancing the data using SMOTE...")
sm = SMOTE(random_state=42)

# Define the Random Forest Classifier and MLPClassifier
rfc = RandomForestClassifier(random_state=42)
mlp = MLPClassifier(random_state=42)

# Define the pipeline with SMOTE, normalization and the classifier
pipeline_rfc = Pipeline([('sm', sm), ('scaler', scaler), ('rfc', rfc)])
pipeline_mlp = Pipeline([('sm', sm), ('scaler', scaler), ('mlp', mlp)])

# Define the hyperparameters to tune with GridSearchCV
param_grid_rfc = {'rfc__n_estimators': [50, 100, 200],
                  'rfc__max_depth': [5, 10, 15],
                  'rfc__min_samples_split': [2, 5, 10]}

param_grid_mlp = {'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
                  'mlp__alpha': [0.0001, 0.001, 0.01],
                  'mlp__max_iter': [100, 200, 300]}

# Train the models using GridSearchCV to find the best hyperparameters
print("Training the models using GridSearchCV...")
grid_rfc = GridSearchCV(pipeline_rfc, param_grid_rfc, cv=5, n_jobs=-1)
grid_rfc.fit(X_train, y_train)

grid_mlp = GridSearchCV(pipeline_mlp, param_grid_mlp, cv=5, n_jobs=-1)
grid_mlp.fit(X_train, y_train)


# Print the best hyperparameters for both models
print("Best hyperparameters for Random Forest:", grid_rfc.best_params_)
print("Best hyperparameters for MLP:", grid_mlp.best_params_)

# Predict on the test set using the best models
y_pred_rfc = grid_rfc.predict(X_test)
y_pred_mlp = grid_mlp.predict(X_test)

# Print the classification reports for both models
print("Classification report for Random Forest:")
print(classification_report(y_test, y_pred_rfc))

print("Classification report for MLP:")
print(classification_report(y_test, y_pred_mlp))

# Calculate the ROC curve for both models
y_proba_rfc = grid_rfc.predict_proba(X_test)[:, 1]
fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y_test, y_proba_rfc)

y_proba_mlp = grid_mlp.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, y_proba_mlp)

# Plot the ROC curve for both models
plt.figure(figsize=(8, 8))
plt.plot(fpr_rfc, tpr_rfc, label='Random Forest')
plt.plot(fpr_mlp, tpr_mlp, label='MLP')
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
