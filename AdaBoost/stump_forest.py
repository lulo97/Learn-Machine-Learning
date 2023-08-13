from stump import *
from csv import reader
random.seed(1)

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

class Stumps:
    def __init__(self, feature_data, target_data) -> None:
        self.feature_data = feature_data
        self.target_data = target_data
        self.size = len(feature_data)
        self.feature_size = len(feature_data[0])
        self.forest = []
    
    def findBestNumericThreshold(self, numeric_column_index):
        for i in range(self.size):
            threshold_value = self.feature_data[i][numeric_column_index]
            stump = Stump([threshold_value for _ in range(self.size)])
            stump.makeLeaf(self.target_data, [self.feature_data[_][numeric_column_index] for _ in range(self.size)])
        return threshold_value
    
    def chooseBestStump(self):
        stump_index = 0
        best_stump = None
        while stump_index != self.feature_size:
            if isNum(self.feature_data[0][stump_index]):
                threshold_value = self.findBestNumericThreshold(stump_index)
                stump = Stump([threshold_value for _ in range(self.size)])
                stump.feature_index = stump_index
                stump.makeLeaf(self.target_data, [self.feature_data[_][stump_index] for _ in range(self.size)])
                stump.updateWeight()
            else:
                threshold_value = [row[stump_index] for row in self.feature_data]
                stump = Stump(threshold_value)
                stump.feature_index = stump_index
                stump.makeLeaf(target_data)
                stump.updateWeight()
            if best_stump == None:
                best_stump = stump
            elif stump.gini < best_stump.gini:
                best_stump = stump
            stump_index += 1
        return best_stump

    def getRandomNormIndex(self, weight: list):
        random_norm_indexs = [random.uniform(0, 1) for _ in range(self.size)]
        #random_norm_indexs = [0.72, 0.42, 0.83, 0.51, 0.72, 0.42, 0.83, 0.51]
        norm_weight = [ele + sum(weight[0:i]) for i, ele in enumerate(weight)]
        random_sample_indexs = []
        for num in random_norm_indexs:
            for weight in norm_weight:
                if num < weight:
                    random_sample_indexs.append(norm_weight.index(weight))
                    break
        return random_sample_indexs

    def newDataByNormalizeWeight(self, weight: list):
        random_sample_indexs = self.getRandomNormIndex(weight)
        new_feature_data = [self.feature_data[i] for i in random_sample_indexs]
        new_target_data = [self.target_data[i] for i in random_sample_indexs]
        return new_feature_data, new_target_data

    def predict(self, new_row):
        predict_output = {"Yes": 0, "No": 0}
        for stump in self.forest:
            predict_output[stump.predict(new_row)]
            predict_output[stump.predict(new_row)] += stump.amount_of_say
        return max(predict_output, key=predict_output.get)

    def accuracy(self, feature_data_test, target_data_test):
        predicts = [self.predict(target) for target in feature_data_test]
        return sum([predict == actual for predict, actual in zip(predicts, target_data_test)])/len(target_data_test)

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

    feature_data = [row[:-1] for row in data]
    target_data = [row[-1] for row in data]

    stumps = Stumps(feature_data, target_data)

    #Loop tạo n stump
    for _ in range(10):
        best_stump = stumps.chooseBestStump()
        stumps.forest.append(best_stump)
        stumps.feature_data, stumps.target_data = stumps.newDataByNormalizeWeight(best_stump.weight)

    for ele in stumps.forest:
        continue
        print(ele, "Thuộc tính", ele.feature_index, "Threshold", ele.threshold)

    print(stumps.accuracy(feature_data, target_data))