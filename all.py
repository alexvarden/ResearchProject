
import warnings
from NN_Classifier import *
from NN_Regressor import *
from CounterfactualSurrogateModel import *

 
warnings.filterwarnings(
    "ignore", message="DataFrame is highly fragmented")




# # # -------------------------------

# iris_category_features = []
# iris_continous_features = ['sepal_length','sepal_width','petal_length','petal_width']

# iris = NN_Classifier('iris', hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100),
#     categorical_features=iris_category_features, 
#     continous_features=iris_continous_features
# )

# iris.load_data()
# iris.split_data()
# # iris.train()
# iris.loadModel()
# iris.evaluate() 

# example = iris.X.iloc[53]
# localisedData = iris.getLocalisedData(50, example)

# print(localisedData)

# iris_surrogateModel = CountefactualSurrogateModel('iris',
#     categorical_features=iris.categorical_features,
#     continous_features=iris.continous_features,
#     className="class",
#     fileModifer="local",
#     regression=False,
#     n_samples=3    
# )
# iris_surrogateModel.loadData()
# iris_surrogateModel.loadModel(iris.clf)


# iris_surrogateModel.generate(scale=1, localisedData=localisedData)
# iris_surrogateModel.loadData(path=f"counterfactuals/{iris.name}-1.csv")
# iris_surrogateModel.n_samples = 1
# iris_surrogateModel.generate(scale=1, generation=2)
# iris_surrogateModel.generateTree(dataset=[1,2])


# # #  # -------------------------------

dryBean_category_features = []
dryBean_continous_features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea',
                              'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

dryBean = NN_Classifier('dry-bean', 
    hidden_layer_sizes=(16, 50, 50, 50, 50, 50, 50, 50, 50, 1000),
    categorical_features=dryBean_category_features,
    continous_features=dryBean_continous_features
)

dryBean.load_data()
dryBean.split_data()
# dryBean.train()
dryBean.loadModel()
dryBean.evaluate()

example = dryBean.X.iloc[[66]]
localisedData = dryBean.getLocalisedData(example, 0.2)
localisedClasses = dryBean.getLocalisedData(example, 0.3)

print("Localised samples :", len(localisedData))

print(localisedData['class'])
print(localisedClasses['class'])

# localisedData, localisedValidationData = train_test_split(
#     localisedData, test_size=0.2, random_state=1)

# dryBean_surrogateModel = CountefactualSurrogateModel('dry-bean',
#     categorical_features=dryBean.categorical_features,
#     continous_features=dryBean.continous_features,
#     className="class",
#     fileModifer="local",
#     n_samples=1
# )


# dryBean_surrogateModel.loadModel(dryBean.clf)
# dryBean_surrogateModel.loadDataSet(localisedClasses)
# dryBean_surrogateModel.generate(scale=1, localisedData=localisedData)
# dryBean_surrogateModel.n_samples = 1
# dryBean_surrogateModel.generate(
#     scale=1, localisedData=pd.read_csv(f"counterfactuals/{dryBean.name}-1.csv"), generation=2)
# dryBean_surrogateModel.generateTree(
#     dataset=[1,2], localisedData=localisedData,
#     localisedValidationData=localisedValidationData)

# # # # -------------------------------

# category_features = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex', 'native-country']
# continous_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week','education-num']
# date_features = []


# adult = NN_Classifier('adult',
#     hidden_layer_sizes=(100, 100, 100, 100),
#     categorical_features=category_features, continous_features=continous_features
# )
# adult.load_data()
# adult.split_data()
# # adult.train()
# adult.loadModel()
# adult.evaluate()


# example = adult.X.iloc[[222]]
# localisedData = adult.getLocalisedData(example, 2)


# print(localisedData)


# print("Localised samples :", len(localisedData))

# localisedData, localisedValidationData = train_test_split(
# localisedData, test_size=0.2, random_state=1)

# adult_surrogateModel = CountefactualSurrogateModel('adult',
#     categorical_features=adult.categorical_features,
#     continous_features=adult.continous_features,
#     className="class",
#     fileModifer="local",
#     n_samples=1
# )
# adult_surrogateModel.loadModel(adult.clf)
# adult_surrogateModel.generate(scale=1, localisedData=localisedData)
# adult_surrogateModel.n_samples = 1
# adult_surrogateModel.generate(
# scale=1, localisedData=pd.read_csv(f"counterfactuals/{adult.name}-1.csv"), generation=2)
# adult_surrogateModel.generateTree(
# dataset=[1,2], 
# localisedData=localisedData,
# localisedValidationData=localisedValidationData)


# # # # -------------------------------

# bike_category_features = [ 'weathersit']
# bike_continous_features = ['holiday', 'weekday', 'workingday', 'season',
#                       'yr', 'mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed']

# bike = NN_Regressor('bike-sharing-hourly',
#     hidden_layer_sizes=(100, 100, 100, 100),
#     categorical_features=bike_category_features, 
#     continous_features=bike_continous_features,
#     classname='cnt'
# )
# bike.load_data()
# bike.split_data()
# # bike.train()
# bike.loadModel()
# bike.evaluate()


# example = bike.X.iloc[[1500]]
# localisedData = bike.getLocalisedData(example, 0.1)

# print("Localised samples :",len(localisedData))

# localisedData, localisedValidationData = train_test_split(
#     localisedData, test_size=0.2, random_state=1)


# bike_surrogateModel = CountefactualSurrogateModel('bike-sharing-hourly',
#     categorical_features=bike.categorical_features,
#     continous_features=bike.continous_features,
#     className=bike.classname,
#     regression=True
# )
# bike_surrogateModel.loadModel(bike.clf)
# bike_surrogateModel.generate(scale=1, localisedData=localisedData)
# bike_surrogateModel.generateTree()
# bike_surrogateModel.generate(scale=1, localisedData=pd.read_csv(f"counterfactuals/{bike.name}-1.csv"), generation=2)
# bike_surrogateModel.generateTree(dataset=[1, 2], localisedValidationData=localisedValidationData)


# # # # -------------------------------

# category_features = []
# continous_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "is_red",
#                       "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

# wine = NN_Regressor('wine-quality',
#     hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 10, 10, 10,10),
#     categorical_features=category_features,
#     continous_features=continous_features,
#     classname="quality"
# )
# wine.load_data()
# wine.split_data()
# # wine.train()
# wine.loadModel()
# wine.evaluate()

# wine_surrogateModel = CountefactualSurrogateModel('wine-quality',
#     categorical_features=wine.categorical_features,
#     continous_features=wine.continous_features,
#     className=wine.classname,
#     regression=True,
#     fileModifer = "local"
# )
# wine_surrogateModel.loadModel(wine.clf)
# # wine_surrogateModel.generate(scale=0.1)
# wine_surrogateModel.generateTree()




# # # --------
# # category_features = []
# # continous_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
# #                       "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

# # regressor = NN_Regressor('wine-quality-red',
# #     hidden_layer_sizes=(1000, 1000, 100, 100, 100, 100, 100, 100, 10, 10, 10,10),
# #     categorical_features=category_features,
# #     continous_features=continous_features,
# #      classname="quality",
# #      fileModifer = "local"
# # )
# # regressor.load_data()
# # regressor.split_data()
# # # regressor.train()
# # regressor.loadModel()
# # regressor.evaluate()


# # regressor = NN_Regressor('wine-quality-white',
# #     hidden_layer_sizes=(500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 10, ),
# #     categorical_features=category_features,
# #     continous_features=continous_features,
# #      classname="quality"
# # )
# # regressor.load_data()
# # regressor.split_data()
# # # regressor.train()
# # regressor.loadModel()
# # regressor.evaluate()





