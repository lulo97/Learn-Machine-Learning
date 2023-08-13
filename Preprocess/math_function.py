import math

def getAccuracy(predicted_Y, actual_Y):
    correct = 0
    for i, ele in actual_Y:
        if predicted_Y[i] == actual_Y[i]:
            correct += 1
    return correct

def mean(values):
    if isinstance(values[0], list):
        output = list(map(list, zip(*values)))
        for i, ele in enumerate(output):
            output[i] = sum(output[i])/len(output[i])
        return output
    return sum(values)/len(values)

def RMSE(predicted_Y, actual_Y):
    if len(predicted_Y) == 0:
        return
    output = 0
    for i, ele in enumerate(predicted_Y):
        output += (actual_Y[i] - predicted_Y[i])**2
    return (output/len(predicted_Y))**0.5