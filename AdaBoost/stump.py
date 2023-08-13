import math
import sys
import random

def isNum(value):
    try:
        float(value)
        return True
    except:
        return False

class Leaf:
    def __init__(self, true_data, false_data) -> None:
        self.true_data = true_data
        self.false_data = false_data
        self.true_data_size = len(true_data)
        self.false_data_size = len(false_data)
        self.total = len(true_data) + len(false_data)
        self.computeGini()

    def computeGini(self):
        if self.total == 0:
            self.gini = 1 - 1e-10
        else:
            self.gini = 1 - (self.true_data_size/self.total)**2 - (self.false_data_size/self.total)**2

    def __repr__(self) -> str:
        return f"\n\tTrue = {self.true_data}, {self.true_data_size}\n\tFalse = {self.false_data}, {self.false_data_size}"

class Stump:
    def __init__(self, threshold: list) -> None:
        self.feature_index = None
        self.threshold = threshold
        self.left_leaf = None
        self.right_leaf = None
        self.total_error = None
        self.amount_of_say = None
        self.weight = None

    def makeLeaf(self, target_data, feature_column = None):
        self.weight = [1/len(target_data) for _ in target_data]
        dummy_column = self.threshold

        if isNum(self.threshold[0]) and feature_column != None:
            dummy_column = ["Yes" if float(cell) > float(self.threshold[0]) else "No" for cell in feature_column]

        TP, FP, TN, FN = 0, 0, 0, 0
        TP_index, TN_index, FP_index, FN_index = [], [], [], []
        for i, (cell, target) in enumerate(zip(dummy_column, target_data)):
            if "Yes" == cell == target: TP_index.append(i)
            if "No" == cell == target: TN_index.append(i)
            if "Yes" == cell != target: FP_index.append(i)
            if "No" == cell != target: FN_index.append(i)

        self.left_leaf = Leaf(TP_index, FP_index)
        self.right_leaf = Leaf(TN_index, FN_index)
        self.computGini()

    def computGini(self):
        left_gini = self.left_leaf.gini
        right_gini = self.right_leaf.gini
        total = self.left_leaf.total + self.right_leaf.total
        if total == 0:
            self.gini = 1 - 1e-10
        else:
            self.gini = (self.left_leaf.total/total)*left_gini + (self.right_leaf.total/total)*right_gini

    def updateWeight(self):
        false_predict_indexs = self.left_leaf.false_data + self.right_leaf.false_data
        self.total_error = sum([self.weight[i] for i in false_predict_indexs])

        if self.total_error == 0:
            self.total_error += 1e-10
        if self.total_error == 1:
            self.total_error -= 1e-10
        self.amount_of_say = 0.5*math.log((1 - self.total_error)/self.total_error)
        new_weight = [self.weight[0]*math.e**(-self.amount_of_say) for _ in self.weight] 

        for i in false_predict_indexs:
            new_weight[i] = self.weight[i]*math.e**self.amount_of_say

        norm_new_weigh = [ele/sum(new_weight) for ele in new_weight]
        self.weight = norm_new_weigh

    def predict(self, new_row):
        cell = new_row[self.feature_index]
        if isNum(cell):
            return "Yes" if float(cell) > float(self.threshold[0]) else "No"
        return cell

    def __repr__(self) -> str:
        print("Left Leaf: ", self.left_leaf)
        print("Right Leaf: ", self.right_leaf)
        print("Gini: ", self.gini)
        print("Weight: ", self.weight)
        print("Amount of say: ", self.amount_of_say)
        return ""

if __name__ == "__main__":
    data = [
        ['Yes', 'Yes', 205, 'Yes'],
        ['No', 'Yes', 180, 'Yes'],
        ['Yes', 'No', 210, 'Yes'],
        ['Yes', 'Yes', 167, 'Yes'],
        ['No', 'Yes',156, 'No'],
        ['No', 'Yes', 125, 'No'],
        ['Yes', 'No', 168, 'No'],
        ['Yes', 'Yes', 172, 'No']
    ]
    data = [['No', 225, 'Yes'], ['No', 213, 'Yes'], ['No', 124, 'No'], ['Yes', 245, 'Yes'], ['Yes', 203, 'Yes'], ['Yes', 200, 'Yes'], ['No', 134, 'Yes'], ['Yes', 191, 'Yes'], ['No', 225, 'Yes'], ['No', 168, 'No']]
    feature_data = [row[:-1] for row in data]
    target_data = [row[-1] for row in data]

    s1 = Stump([row[0] for row in feature_data])
    s1.makeLeaf(target_data)
    s1.updateWeight()
    s1.feature_index = 0
    print(s1)
    print("Predict", [s1.predict(ele) for ele in feature_data])

    s3 = Stump([168 for row in feature_data[0]])
    s3.makeLeaf(target_data, [row[1] for row in feature_data])
    s3.updateWeight()
    s3.feature_index = 1
    print(s3)
    print("Predict", [s3.predict(ele) for ele in feature_data])




"""
Một node chứa question dạng x = threshold hoặc x > threshold?
- threshold = [threshold_i với i = 1, m] = ["Yes", "No", ...] là cột của thuộc tính nào đó
- threshold = 123 là giá trị số
- Trả lời x == "Yes"? hoặc x > 123? để vẽ ra hai lá là lá phải (1 - đúng) và lá sai (0 - sai)

Lá trái:
- Chứa TP, FP
- Chỉ số các hàng TP, FP

Lá phải:
- Chứa TN, FN
- Chỉ số các hàng TN, FN

Tính các chỉ số trong lá cần cột của bảng (feature_index, feature_data)

"""