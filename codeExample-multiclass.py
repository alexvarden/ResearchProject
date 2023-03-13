trainingData= 1

counterFactuals = []
classes = ["Iris-versicolor","Iris-setosa","Iris-virginica"]

percentageToSample = 1 / (len(classes) - 1)

for currentClass in classes:
    queryInstances = trainingData.isNotClass(currentClass)
    for oppositeClass in classes:
        if (oppositeClass == currentClass):
            continue
        
        data = queryInstances
        .sample(frac=percentageToSample, stratified="class")
        .drop(['class'], axis=1)
        
        counterfactual= dice.generate_counterfactuals(data,desired_class=oppositeClass)
        counterFactuals.add(counterfactual)



