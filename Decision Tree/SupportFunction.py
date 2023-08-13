from Question import Question

def getUniqueOfData(data, col):
    return set([row[col] for row in data])

def countUniqueOfData(data):
    count_dict = {}
    for row in data:
        label = row[-1]
        if label not in count_dict:
            count_dict[label] = 0
        count_dict[label] += 1
    return count_dict

def partition(data: list, question: Question):
    true_data, false_data = [], []
    for row in data:
        if question.match(row):
            true_data.append(row)
        else:
            false_data.append(row)
    return true_data, false_data

def giniOfData(data: list):
    count_dict = countUniqueOfData(data)
    impurity = 1
    for label in count_dict:
        percent_of_label = count_dict[label]/len(data)
        impurity -= percent_of_label**2
    return impurity

def infomationGain(left: list, right: list, parent_gini: float):
    total_len = len(left) + len(right)
    average_gini = (giniOfData(left)*len(left) + giniOfData(right)*len(right))/total_len
    return parent_gini - average_gini

def findBestPartition(data: list, header: list):
    best_gain, best_question = 0, None
    current_gini = giniOfData(data)
    number_of_features  = len(data[0]) - 1

    for colmn in range(number_of_features):
        unique_value = getUniqueOfData(data, colmn)
        for value in  unique_value:
            question = Question(colmn, value, header)
            true_data, false_data = partition(data, question)
            if len(true_data) == 0 or len(false_data) == 0:
                continue
            gain = infomationGain(true_data, false_data, current_gini)
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


if __name__ == "__main__":
    header = ["like", "age", "dislike"]
    data = [
        ["apple", 12, "grape"],
        ["lemon", 8, "apple"],
        ["apple", 14, "melon"],
        ["mango", 5, "apple"],
        ["coconut", 7, "apple"],
    ]
    true_data, false_data = partition(data, Question(0, "apple", header))
    #print(true_data)
    #print(false_data)

    test_gini = [['Apple'], ['Orange']]
    test_gini2 = [['Apple'],
                  ['Orange'],
                  ['Grape'],
                  ['Grapefruit'],
                  ['Blueberry']]
    test_gini3 = [['Apple'], ['Apple']]
    print(giniOfData(test_gini3))