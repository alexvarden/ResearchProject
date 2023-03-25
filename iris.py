from NN_Classifier import *
from NN_Regressor import *
from CounterfactualSurrogateModel import *
import time


N_SAMPES = 1500


# # # -------------------------------

iris_category_features = []
iris_continous_features = ['sepal_length',
                           'sepal_width', 'petal_length', 'petal_width']

iris = NN_Classifier('iris', hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100),
                     categorical_features=iris_category_features, continous_features=iris_continous_features)

iris.load_data()
iris.split_data()
# iris.train()
iris.loadModel()
iris.evaluate()

start_time = time.time()

print("STARTING GENERATION 1")
iris_data = iris.getGlobalRandomSample(n_samples=N_SAMPES)

iris_surrogateModel = CountefactualSurrogateModel('iris',
    categorical_features=iris.categorical_features,
    continous_features=iris.continous_features,
    className="class",
    regression=False,
    n_samples=1
)
iris_surrogateModel.loadModel(iris.clf)
iris_surrogateModel.loadDataSet(iris_data)
# iris_surrogateModel.generate(scale=1)


print("STARTING GENERATION 2")
iris_surrogateModel.loadData(
    path=f"counterfactuals/{iris_surrogateModel.fileModifer}-{iris.name}-1.csv")
iris_surrogateModel.n_samples = 1
# iris_surrogateModel.generate(scale=1, generation=2)

print("CounterFactual")
iris_surrogateModel.generateTree(dataset=[1, 2])

print("Control")
iris_surrogateModel.generateTree(
    localisedData=iris_data)

end_time = time.time()
print("Elapsed time: ", end_time - start_time, "seconds")
print(f"SampleSize : {N_SAMPES}")
