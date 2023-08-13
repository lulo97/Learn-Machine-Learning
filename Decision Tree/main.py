from Leaf import Leaf
from DecisionNode import DecisionNode
from SupportFunction import *

import random
import sys
import os

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../Preprocess")

from split_data import SplitData
from read_file import *
from filter_function import *

def buildTree(data: list, header: list):
    info, question = findBestPartition(data, header)
    if info == 0:
        return Leaf(data)
    true_data, false_data = partition(data, question)
    true_branch = buildTree(true_data, header)
    false_branch = buildTree(false_data, header)
    return DecisionNode(question, true_branch, false_branch)

def printTree(node, spacing = ""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.prediction)
        return
    
    print(spacing + str(node.question))

    print (spacing + '--> True:')
    printTree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    printTree(node.false_branch, spacing + "  ")

def runClassify(row, node):
    if isinstance(node, Leaf):
        return node.prediction
    if node.question.match(row):
        return runClassify(row, node.true_branch)
    else:
        return runClassify(row, node.false_branch)

def printLeaf(count_unique: dict):
    total_uniques = sum(count_unique.values())
    unique_proportion = {}
    for label in count_unique.keys():
        unique_proportion[label] = f"{count_unique[label]/total_uniques * 100}%"
    return unique_proportion 

def getAcurracy(test_data, tree):
    accuracy = 0
    for row in test_data:
        predict = runClassify(row, tree)
        random_predict = random.choices(list(predict.keys()), list(predict.values()))
        actual = row[-1]
        #print(random_predict, actual)
        if actual == random_predict[0]:
            accuracy += 1
    return accuracy/len(test_data) * 100

if __name__ == "__main__":
    """
    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    header = ["color", "diameter", "label"]
    tree = buildTree(training_data, header)
    printTree(tree)

    # Evaluate
    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    accuracy = getAcurracy(testing_data, tree)
    print(accuracy)
    """

    csv_dir = r"iris.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)
    split = SplitData(data)

    header = [f"X{i}" for i in range(len(data[0]) - 1)] + ["label"]

    trains, tests = split.crossValidation(5)
    for i, train in enumerate(trains):
        #[print(ele) for ele in train]
        #print(train[-10:-1], len(tests[i]))
        tree = buildTree(train, header)
        print(getAcurracy(tests[i], tree))
