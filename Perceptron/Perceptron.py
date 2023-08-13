from SplitData import *
import math

class Perceptron:
    def __init__(self, X, Y, X_test, Y_test) -> None:
        self.X, self.Y = X, Y
        self.X_test, self.Y_test = X_test, Y_test
        self.X_transpose = list(map(list, zip(*self.X)))
        self.size = len(X)
        self.b1 = [0 for _ in X[0]]
        self.b0 = 0

    def dot(self, vector_a: list, vector_b: list):
        return sum([ai*bi for ai, bi in zip(vector_a, vector_b)])
    
    def linear_term(self, x):
        return self.dot(x, self.b1) + self.b0

    def activation(self, x):
        if self.linear_term(x) >= 0:
            return 1
        return 0

    def deltaB1(self, x, y, y_pre):
        sign = y - y_pre
        return [sign * xi for xi in x]
    
    def deltaB0(self, x, y, y_pre):
        return y - y_pre
    
    def training(self, lr, step):
        for _ in range(step):
            for i, x in enumerate(self.X):
                y_pre = self.activation(x)
                E1 = self.deltaB1(self.X[i], self.Y[i], y_pre)
                self.b0 += lr*self.deltaB0(self.X[i], self.Y[i], y_pre)
                self.b1 = [self.b1[i] + lr*eb for i, eb in enumerate(E1)]
            if _ % 10 == 1:
                self.accuracyScore()

    def accuracyScore(self):
        score = 0
        Y_predicted = [self.activation(ele) for ele in self.X_test]
        for i, ele in enumerate(Y_predicted):
            if  ele == self.Y_test[i]:
                score += 1
        print("Accuracy: ", score/len(X_test) * 100)

if __name__ == "__main__":
    csv_dir = r"C:\Users\admin\Desktop\Đồ án 1\Data\seeds_dataset.csv"
    #csv_dir = r"C:\Users\admin\Desktop\Đồ án 1\Data\pima-indians-diabetes.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)

    split = SplitData(data)
    X_train, Y_train, X_test, Y_test = split.splitting(0.9)

    P = Perceptron(X_train, Y_train, X_test, Y_test)
    P.training(1e-1, 10000)
    print(P.b1, P.b0)


    