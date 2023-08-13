from SplitData import *
import math

class LogisticRegression:
    def __init__(self, X, Y, X_test, Y_test) -> None:
        self.X, self.Y = X, Y
        self.X_test, self.Y_test = X_test, Y_test
        self.X_transpose = list(map(list, zip(*self.X)))
        self.size = len(X)
        self.b1 = [0 for _ in X[0]]
        self.b0 = 0
        self.max_x, self.min_x = max(map(max, X)), min(map(min, X))
        
    def dot(self, vector_a: list, vector_b: list):
        return sum([ai*bi for ai, bi in zip(vector_a, vector_b)])

    def linear_term(self, x):
        return self.dot(self.b1, self.norm(x)) + self.b0

    def roundPredict(self, x):
        return 1 if self.predict(x) >= 0.5 else 0

    def predict(self, x):
        return 1/(1 + math.e**-self.linear_term(x))
    
    def norm(self, x):
        return [(xi - self.min_x)/(self.max_x - self.min_x) for xi in x]

    def accuracyScore(self):
        score = 0
        Y_predicted = [self.roundPredict(ele) for ele in self.X_test]
        for i, ele in enumerate(Y_predicted):
            if  ele == self.Y_test[i]:
                score += 1
        print("Accuracy: ", score/len(X_test) * 100)

    def gradientB1(self, x, y):
        predict_y = self.predict(x)
        temp = (y - predict_y)*predict_y*(1 - predict_y)
        return [temp * xi for xi in x]

    def gradientB0(self, x, y):
        predict_y = self.predict(x)
        return (y - predict_y)*predict_y*(1 - predict_y)
    
    def gradientB1_ver2(self, x, y):
        temp = self.predict(x) - y
        return [- temp * xi for xi in x]

    def gradientB0_ver2(self, x, y):
        return -(self.predict(x) - y)

    def gradientMethod(self, lr, step):
        for _ in range(step):
            E1, E0 = [0 for _ in self.X[0]], 0
            for i, x in enumerate(self.X):
                E0 += self.gradientB0_ver2(self.X[i], self.Y[i])
                E1 = [eb + ele for eb, ele in zip(E1, self.gradientB1_ver2(self.X[i], self.Y[i]))]
            self.b0 += lr*E0
            self.b1 = [self.b1[i] + lr*eb for i, eb in enumerate(E1)]
            
            if _ % 1000 == 1:
                lr = lr * 0.9
                self.accuracyScore()

if __name__ == "__main__":
    file_name = "seeds_dataset.csv"
    csv_dir = rf"C:\Users\admin\Desktop\Đồ án 1\Data\{file_name}"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)

    split = SplitData(data)
    X_train, Y_train, X_test, Y_test = split.splitting(0.9)

    LR = LogisticRegression(X_train, Y_train, X_test, Y_test)
    LR.gradientMethod(0.1, 100000)
    print(LR.b1, LR.b0)