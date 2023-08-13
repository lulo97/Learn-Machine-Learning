import random
from activation import Activation
from matrix_operation import *
from csv import reader

class NeuralNetwork:
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
        self.alpha = 0.01
        self.initialParameter()

    def initialParameter(self):
        self.m = len(self.X)
        self.n = len(self.X[0])
        self.k = 10
        self.l = len(self.Y[0])
        self.w1 = [[random.random() for i in range(self.n)] for j in range(self.k)]
        self.w2 = [[random.random() for i in range(self.k)] for j in range(self.l)]
        self.output = self.w2
    
    def predict(self, x: list[float]):
        hidden_layer = Activation.computeMatrix(Mutiply.matrix([x], self.w1))
        output = Activation.computeMatrix(Mutiply.matrix(hidden_layer, self.w2))
        return output[0]
    
    def forward(self):
        XdotW1 = Mutiply.matrix(self.X, self.w1)
        self.hidden_layer = Activation.computeMatrix(XdotW1)
        Layer1dotW2 = Mutiply.matrix(self.hidden_layer, self.w2)
        self.output = Activation.computeMatrix(Layer1dotW2)

    def getGradientW2(self):
        y_minus_yhat_x2 = Mutiply.constToMatrix(2, Subtract.matrix(self.Y, self.output))
        deri_sigmoid_output = Activation.derivativeMatrix(self.output)
        cache = Mutiply.eleByEleMatrix(y_minus_yhat_x2, deri_sigmoid_output)
        hidden_layer_T = transpose(self.hidden_layer)
        return cache, Mutiply.matrix(hidden_layer_T, cache)
        
    def getGradientW1(self, cache):
        cache_dot_w2T = Mutiply.matrix(cache, transpose(self.w2))
        deri_sigmoid_layer1 = Activation.derivativeMatrix(self.hidden_layer)
        cache2 = Mutiply.eleByEleMatrix(cache_dot_w2T, deri_sigmoid_layer1)
        X_transpose = transpose(self.X)
        return Mutiply.matrix(X_transpose, cache2)

    def backward(self):
        cache, dw2 = self.getGradientW2()
        dw1 = self.getGradientW1(cache)
        self.w1 = Add.matrix(self.w1, Mutiply.constToMatrix(self.alpha, dw1))
        self.w2 = Add.matrix(self.w2, Mutiply.constToMatrix(self.alpha, dw2))

    def error(self, X_test, Y_test):
        Y_predict = []
        for x in X_test:
            Y_predict.append(self.predict(x))
        return RMSEMatrix(Y_test, Y_predict)

    def roundPercentPredict(self, x:list):
        round_predict = [0 for _ in self.predict(x)]
        round_predict[self.predict(x).index(max(self.predict(x)))] = 1
        return round_predict

    def accuracy(self, X_test, Y_test):
        score = 0
        for x, y in zip(X_test, Y_test):
            if self.roundPercentPredict(x) == y:
                score += 1
        return score/len(X_test)

def loadDataFromCSV(csv_dir: str):
    with open(csv_dir, 'r', encoding = 'utf-8-sig') as file:
        content = reader(file)
        return list(content)
    
def convertToFloat(data):
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            try:
                data[i][j] = float(col)
            except:
                continue
    return data

def oneHot(Y):
    new_y = []
    category_to_num = list(set(Y))
    dict_y = {category_to_num[i]: i for i in range(len(category_to_num))}
    for y in Y:
        cache = [0]*len(category_to_num)
        cache[dict_y[y]] = 1
        new_y.append(cache)
    return new_y

def normData(X: list[list]):
    X_norm = []
    X_transpose = list(map(list, zip(*X)))
    mins = [min(column) for column in X_transpose]
    maxs = [max(column) for column in X_transpose]
    for i, column in enumerate(X_transpose):
        try:
            norm_column = [(cell - mins[i])/(maxs[i] - mins[i]) for cell in column]
        except:
            norm_column = [cell/max(maxs) for cell in column]
        X_norm.append(norm_column)
    return list(map(list, zip(*X_norm)))

if __name__ == "__main__":       
    data = loadDataFromCSV("data_banknote_authentication.csv")
    data = convertToFloat(data)

    X = [row[:-1] for row in data]
    X = normData(X)
    Y = [row[-1] for row in data]
    Y_onehot = oneHot(Y)

    nn = NeuralNetwork(X, Y_onehot)
    nn.k = 10
    nn.alpha = 0.01

    for i in range(1000):
        nn.forward()
        nn.backward()
        if i % 99 == 1:
            print(f"RMSE = {nn.error(X, Y_onehot)}, Accuracy = {nn.accuracy(X, Y_onehot)}")

    if 0:
        x1 = [5.1,3.5,1.4,0.2]
        x2 = [7.0,3.2,4.7,1.4]
        x3 = [6.3,3.3,6.0,2.5]

        for x in [x1, x2, x3]:
            print(x, nn.predict(x), nn.roundPercentPredict(x))

        print(nn.accuracy(X, Y_onehot))