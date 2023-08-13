import random
import sys
import os

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../Preprocess")

from split_data_feature_target import SplitDataFeatureTarget
from read_file import *
from filter_function import *
from math_function import *

class StochasticGD:
    def __init__(self, X, Y, X_test, Y_test) -> None:
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.size = len(X)
        self.X_transpose = list(map(list, zip(*self.X)))
        self.b1 = [0 for _ in self.X[0]]
        self.b0 = 0
    
    def predict(self, x_value):
        temp = 0
        for i, ele in enumerate(self.b1):
            temp += ele * x_value[i]
        return temp + self.b0
    
    def evaluate_error(self):
        Y_predicted = [self.predict(ele) for ele in self.X_test]
        RMSE_score = RMSE(Y_predicted, self.Y_test) 
        print("Error: ", RMSE_score)

    def fit(self, learning_rate, step):
        for _ in range(step):
            random_index = random.randrange(0, len(self.X) - 1)
            b0_derivative = -2 * (self.Y[random_index] - self.predict(self.X[random_index]))
            b1_derivative = []
            for j, xij in enumerate(self.X[random_index]):
                b1_derivative.append(-2 * xij * (self.Y[random_index] - self.b1[j] * xij - self.b0))
            self.b0 -= learning_rate * b0_derivative
            self.b1 = [self.b1[i] - learning_rate * b1_deri for i, b1_deri in enumerate(b1_derivative)]
            self.evaluate_error()

if __name__ == "__main__":
    csv_dir = r"BostonHousing.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)
    split = SplitDataFeatureTarget(data)
    
    X_train, Y_train, X_test, Y_test = split.splitting(0.9)
    Xs_train, Ys_train, Xs_test, Ys_test = split.crossValidation(5)

    SGD = StochasticGD(X_train, Y_train, X_test, Y_test)
    SGD.fit(1e-9, 10000)

    Y_predicted = [SGD.predict(ele) for ele in X_test]
    RMSE_score = RMSE(Y_predicted, Y_test)
    print(f"f(x) = {SGD.b1}x + {SGD.b0}")
    print("RMSE = ", RMSE_score)