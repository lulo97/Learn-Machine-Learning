import math
import numpy as np
from SplitData2 import *

def unique(_list: list):
    output = []
    for ele in _list:
        if ele not in output:
            output.append(ele)
    return output

def formatDict(_dict: dict):
    keys = list(_dict.keys())
    keys.sort()
    return {int(i): _dict[i] for i in keys}

#prior_dict = {loại yi: tỷ lệ yi}
def getPrior(data):
    y = [row[-1] for row in data]
    prior_dict = dict(zip(unique(y), [0]*len(unique(y))))
    for row in data:
        prior_dict[row[-1]] += 1
    for key in prior_dict.keys():
        prior_dict[key] /= len(y)
    return formatDict(prior_dict)

def getStandardDeriation(_list, mean):
    sum = 0
    for ele in _list:
        sum += (ele - mean)**2
    return math.sqrt(sum/(len(_list) - 1))

#P(x | y)
def getLikelihoodGaussian(data, feature_number, x, y):
    filter_y_data = []
    for row in data:
        if row[-1] == y:
            filter_y_data.append(row)
    
    mean = sum([row[feature_number] for row in filter_y_data])/len(filter_y_data)
    std = getStandardDeriation([row[feature_number] for row in filter_y_data], mean)

    p_x_given_y = (1 / (math.sqrt(2 * math.pi) * std)) *  math.exp(-((x - mean)**2 / (2 * std**2 )))

    return p_x_given_y

def runNaiveBayes(data_train, data_test):
    X = [row[:-1] for row in data_test]
    features = [i for i in range(len(data_train[0]) - 1)]
    prior = getPrior(data_train)

    y_predict = []

    for row in X:
        likelihood = [1]*len(prior.keys())
        for y_category in prior.keys():
            for feature in features:
                likelihood[y_category] *= getLikelihoodGaussian(data_train, feature, row[feature], y_category)

        prior_predict = {}
        for y_category in prior.keys():
            prior_predict[y_category] = likelihood[y_category] * prior[y_category]

        y_predict.append(max(prior_predict, key = prior_predict.get))

    return y_predict

def predict(data_train, X_test):
    data_test = [ele + [None] for ele in X_test]
    return runNaiveBayes(data_train, data_test)

def accuracy(data_test, y_predict):
    score = 0
    for i, row in enumerate(data_test):
        if row[-1] == y_predict[i]:
            score += 1
    return score/len(data_test)

if __name__ == "__main__":
    csv_dir = r"C:\Users\admin\Desktop\Đồ án 1\Data\seeds_dataset.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)

    split = SplitData2(data)

    trains, tests = split.crossValidation(5)

    #for i, train in enumerate(trains):
        #y_predict = runNaiveBayes(train, tests[i])
        #print(accuracy(tests[i], y_predict))
    
    print(predict(trains[0], [[18.88	,16.26,	0.8969,	6.084	,3.764	,1.649	,6.109, 1]]))



