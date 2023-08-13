import random
import sys
import os

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../Preprocess")

from split_data_feature_target import SplitDataFeatureTarget
from read_file import *
from filter_function import *

def indexOfKSmallest(_list, k):
    return sorted(range(len(_list)), key = lambda sub: _list[sub])[:k]

class KNearest:
    def __init__(self, X, Y, k) -> None:
        self.X = X
        self.Y = Y
        self.k = k
        self.size = len(X)

    def euclideanDistance(self, rowA, rowB):
        sum = 0
        for i, ele in enumerate(rowA):
            sum += (ele - rowB[i])**2
        return sum**0.5

    def findKNearest(self, row0):
        neightbors = []
        dists = []
        for i, row in enumerate(self.X):
            dists.append(self.euclideanDistance(row0, row))
        k_smallest_index = indexOfKSmallest(dists, self.k)
        y_predicts = [self.Y[i] for i in k_smallest_index]
        return y_predicts

    def predict(self, row0):
        y_values = self.findKNearest(row0)
        return max(set(y_values), key = y_values.count)
    
    def accuracy(self, X_test, Y_test):
        score = 0
        for i, x in enumerate(X_test):
            if self.predict(x) == Y_test[i]:
                score += 1
        return score/len(X_test)

if __name__ == "__main__":
    csv_dir = r"iris.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)
    data = removeEmptyRecord(data)

    split = SplitDataFeatureTarget(data)
    X_train, Y_train, X_test, Y_test = split.splitting(0.9)
    Xs_train, Ys_train, Xs_test, Ys_test = split.crossValidation(5)
    
    #print(X_test)

    #knn = KNearest(X_train, Y_train, k = 5)
    #print(knn.accuracy(X_test, Y_test))

    for i, train in enumerate(Xs_train):  
        knn = KNearest(train, Ys_train[i], k = 5)
        print(knn.accuracy(Xs_test[i], Ys_test[i]))


