
from NN_Classifier import *
from NN_Regressor import *
from CounterfactualSurrogateModel import *



N_SAMPES = 1000


# # # -------------------------------

# iris_category_features = []
# iris_continous_features = ['sepal_length','sepal_width','petal_length','petal_width']

# iris = NN_Classifier('iris', hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100),
#                            categorical_features=iris_category_features, continous_features=iris_continous_features)

# iris.load_data()
# iris.split_data()
# iris.train()
# iris.loadModel()
# iris.evaluate() 


# iris_data = iris.getGlobalRandomSample(n_samples=N_SAMPES)

# iris_surrogateModel = CountefactualSurrogateModel('iris',
#     categorical_features=iris.categorical_features,
#     continous_features=iris.continous_features,
#     className="class",
#     regression=False,
#     n_samples=1    
# )
# iris_surrogateModel.loadModel(iris.clf)
# iris_surrogateModel.loadDataSet(iris_data)
# iris_surrogateModel.generate(scale=1)
# iris_surrogateModel.loadData(path=f"counterfactuals/{iris.name}-1.csv")
# iris_surrogateModel.n_samples = 1
# iris_surrogateModel.generate(scale=1, generation=2)
# iris_surrogateModel.generateTree(dataset=[1, 2])


# # #  # -------------------------------


# dryBean_category_features = []
# dryBean_continous_features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea',
#                               'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

# dryBean = NN_Classifier('dry-bean', 
#     hidden_layer_sizes=(16, 50, 50, 50, 50, 50, 50, 50, 50, 1000),
#     categorical_features=dryBean_category_features,
#     continous_features=dryBean_continous_features
# )

# dryBean.load_data()
# dryBean.split_data()
# # dryBean.train()
# dryBean.loadModel()
# dryBean.evaluate()


# dryBean_surrogateModel = CountefactualSurrogateModel('dry-bean',
#     categorical_features=dryBean.categorical_features,
#     continous_features=dryBean.continous_features,
#     className="class",
#     n_samples=1
# )
# dryBean_surrogateModel.loadModel(dryBean.clf)
# # dryBean_surrogateModel.generate(scale=1)
# # dryBean_surrogateModel.loadData(path=f"counterfactuals/{dryBean.name}-1.csv")
# # dryBean_surrogateModel.generate(scale=1, generation=2)
# dryBean_surrogateModel.generateTree(dataset=[1, 2])


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

# adult_surrogateModel = CountefactualSurrogateModel('adult',
#     categorical_features=adult.categorical_features,
#     continous_features=adult.continous_features,
#     className="class"
# )
# adult_surrogateModel.loadModel(adult.clf)
# adult_surrogateModel.generate(scale=1)


# adult_surrogateModel.generate(scale=1)
# adult_surrogateModel.loadData(path=f"counterfactuals/{adult.name}-1.csv")
# adult_surrogateModel.generate(scale=1, generation=2)
# adult_surrogateModel.generateTree(dataset=[1, 2])



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
# bike.train()
# bike.loadModel()
# bike.evaluate()

# bike_surrogateModel = CountefactualSurrogateModel('bike-sharing-hourly',
#     categorical_features=bike.categorical_features,
#     continous_features=bike.continous_features,
#     className=bike.classname,
#     regression=True
# )
# bike_surrogateModel.loadModel(bike.clf)
# bike_surrogateModel.generate(scale=0.1)
# bike_surrogateModel.generateTree()


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
# wine.train()
# wine.loadModel()
# wine.evaluate()

# wine_surrogateModel = CountefactualSurrogateModel('wine-quality',
#     categorical_features=wine.categorical_features,
#     continous_features=wine.continous_features,
#     className=wine.classname,
#     regression=True
# )
# wine_surrogateModel.loadModel(wine.clf)
# wine_surrogateModel.generate(scale=0.1)
# wine_surrogateModel.generateTree()




# # --------
# category_features = []
# continous_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
#                       "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

# regressor = NN_Regressor('wine-quality-red',
#     hidden_layer_sizes=(1000, 1000, 100, 100, 100, 100, 100, 100, 10, 10, 10,10),
#     categorical_features=category_features,
#     continous_features=continous_features,
#      classname="quality"
# )
# regressor.load_data()
# regressor.split_data()
# # regressor.train()
# regressor.loadModel()
# regressor.evaluate()


# regressor = NN_Regressor('wine-quality-white',
#     hidden_layer_sizes=(500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 10, ),
#     categorical_features=category_features,
#     continous_features=continous_features,
#      classname="quality"
# )
# regressor.load_data()
# regressor.split_data()
# # regressor.train()
# regressor.loadModel()
# regressor.evaluate()





