from SplitData import *

class LinearRegression:
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
        self.size = len(X)
        self.mean_x = MathTool.mean(self.X)
        self.mean_y = MathTool.mean(self.Y)
        self.X_transpose = list(map(list, zip(*self.X)))
        self.getVariance()
        self.getCovariance()
        self.b1 = [1 for _ in self.covariance]
        self.b0 = 0
    
    def getVariance(self):
        self.variance = []
        for i, xi in enumerate(self.X_transpose):
            temp = sum([(xij - self.mean_x[i])**2 for xij in xi])
            self.variance.append(temp)
    
    def getCovariance(self):
        self.covariance = []
        for i, xi in enumerate(self.X_transpose):
            temp = 0
            for j, xij in enumerate(xi):
                temp += (xij - self.mean_x[i])*(self.Y[j] - self.mean_y)
            self.covariance.append(temp)

    def getB1(self):
        self.b1 = []
        for i, ele in enumerate(self.covariance):
            self.b1.append(ele/self.variance[i])
    
    def getB0(self):
        temp = 0
        for i, ele in enumerate(self.b1):
            temp += ele * self.mean_x[i]
        self.b0 = self.mean_y - temp
    
    def predict(self, x_value):
        temp = 0
        for i, ele in enumerate(self.b1):
            temp += ele * x_value[i]
        return temp + self.b0

if __name__ == "__main__":
    csv_dir = r"C:\Users\admin\Desktop\Đồ án 1\Data\a.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)

    split = SplitData(data)
    X_train, Y_train, X_test, Y_test = split.splitting(0.9)
    Xs_train, Ys_train, Xs_test, Ys_test = split.crossValidation(5)

    LR = LinearRegression(X_train, Y_train)

    LR.getB1()
    LR.getB0()
    Y_predicted = [LR.predict(ele) for ele in X_test]
    RMSE_score = MathTool.RMSE(Y_predicted, Y_test)
    print(f"f(x) = {LR.b1}x + {LR.b0}")
    print("RMSE = ", RMSE_score)

