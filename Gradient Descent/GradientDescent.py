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

class GradientDescent:
    def __init__(self, X, Y, X_test, Y_test) -> None:
        self.X = X
        self.Y = Y
        self.X_test, self.Y_test = X_test, Y_test
        self.size = len(X)
        self.X_transpose = list(map(list, zip(*self.X)))
        self.b1 = [0.5 for _ in self.X[0]]
        self.b0 = 0.5
    
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
            b0_derivative, b1_derivative = 0, []
            for i, row in enumerate(self.X):
                b0_derivative += -2 * (self.Y[i] - self.predict(row))
            for i, row in enumerate(self.X_transpose):
                temp = sum([-2 * row[j] * (self.Y[j] - self.b1[i] * row[j] - self.b0) for j in range(len(row))])
                b1_derivative.append(temp)

            for i, b1i in enumerate(self.b1):
                self.b1[i] -= b1_derivative[i] * learning_rate
            self.b0 -= b0_derivative * learning_rate

            self.evaluate_error()

if __name__ == "__main__":
    csv_dir = r"BostonHousing.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)

    split = SplitDataFeatureTarget(data)
    X_train, Y_train, X_test, Y_test = split.splitting(0.9)
    Xs_train, Ys_train, Xs_test, Ys_test = split.crossValidation(5)

    GD = GradientDescent(X_train, Y_train, X_test, Y_test)
    GD.fit(1e-10, 5000)

    print(f"f(x) = {GD.b1}x + {GD.b0}")