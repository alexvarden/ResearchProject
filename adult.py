from NN_Classifier import *
from NN_Regressor import *
from CounterfactualSurrogateModel import *
import time

N_SAMPES = 1500

# # # -------------------------------
model_category_features = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex', 'native-country']
model_continous_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week','education-num']


model = NN_Classifier('adult',
    hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50, 50, 100, 100 ),
    categorical_features=model_category_features,
    continous_features=model_continous_features
)

model.load_data()
model.split_data()
# model.train()
model.loadModel()
model.evaluate()

start_time = time.time()

print("STARTING GENERATION 1")
model_data = model.getGlobalRandomSample(n_samples=N_SAMPES)

print(model_data)

model_surrogateModel = CountefactualSurrogateModel('adult',
    fileModifer=f"global-{N_SAMPES}",
    categorical_features=model.categorical_features,
    continous_features=model.continous_features,
    className="class",
    regression=False,
    n_samples=1
)
model_surrogateModel.loadModel(model.clf)
model_surrogateModel.loadDataSet(model_data)
# model_surrogateModel.generate(scale=1)


print("STARTING GENERATION 2")
model_surrogateModel.loadData(
    path=f"counterfactuals/{model_surrogateModel.fileModifer}-{model.name}-1.csv")
model_surrogateModel.n_samples = 1
# model_surrogateModel.generate(scale=1, generation=2)

print("CounterFactual")
model_surrogateModel.generateTree(dataset=[1, 2])

print("Control")
model_surrogateModel.generateTree(
    localisedData=model_data)

end_time = time.time()
print("Elapsed time: ", end_time - start_time, "seconds")
print(f"SampleSize : {N_SAMPES}")
