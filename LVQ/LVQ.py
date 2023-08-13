from SplitData2 import *

def uniqueY(data):
    Y = [row[-1] for row in data]
    output = []
    for y in Y:
        if y not in output:
            output.append(y)
    return output

def sub(a: list, b: list):
    return [ai - bi for ai, bi in zip(a, b)]

def add(a: list, b: list):
    return [ai + bi for ai, bi in zip(a, b)]

def mul(k: float, a: list):
    return [k*ai for ai in a]

class LVQ:
    def __init__(self, train, test = None) -> None:
        self.train = train
        self.test = test
        self.feat_size = len(train[0]) - 1
    
    def initialW(self):
        self.w = []
        for y_unique in uniqueY(self.train):
            self.w.append([0]*self.feat_size + [y_unique])
    
    def euclidDist(self, listA, listB):
        output = 0
        for i in range(self.feat_size):
            output += (listA[i] - listB[i])**2
        return output
    
    def getClosestW(self, row):
        dist = []
        for j, wj in enumerate(self.w):
            dist.append(self.euclidDist(wj, row))
        return dist.index(min(dist))

    def updateW(self, row, lr):
        min_index = self.getClosestW(row)
        y = row[-1]

        w_min = self.w[min_index][:-1]
        y_w = [self.w[min_index][-1]]
        x = row[:-1]

        if self.w[min_index][-1] != y:
            self.w[min_index] = sub(w_min, mul(lr, sub(x, w_min))) + y_w
        else:
            self.w[min_index] = add(w_min, mul(lr, sub(x, w_min))) + y_w
    
    def predict(self, row):
        return self.w[self.getClosestW(row)][-1]

    def accuracy(self):
        score = 0
        for row in self.test:
            if self.predict(row) == row[-1]:
                score += 1
        return score / len(self.test)
    
    def fit(self, lr, step):
        for _ in range(step):
            for row in self.train:
                self.updateW(row, lr)

if __name__ == "__main__":

    csv_dir = r"C:\Users\admin\Desktop\Đồ án 1\Data\ionosphere.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)

    split = SplitData2(data)
    #train, test = split.splitting(0.9)
    trains, tests = split.crossValidation(10)

    lr = 0.001
    step = 1000

    accuracys = []
    for i in range(len(trains)):
        lvq = LVQ(trains[i], tests[i])
        lvq.initialW()
        lvq.fit(lr, step)
        accuracys.append(lvq.accuracy())
    print(accuracys, sum(accuracys)/len(accuracys))
