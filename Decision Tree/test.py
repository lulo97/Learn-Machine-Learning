def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def giniOfData(rows):
    counts = class_counts(rows)
    print(counts)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

test_gini = [['Apple'], ['Orange']]
test_gini2 = [['Apple'],
                ['Orange'],
                ['Grape'],
                ['Grapefruit'],
                ['Blueberry']]
test_gini3 = [['Apple'], ['Apple']]
print(giniOfData(test_gini3))