def convertToFloat(data):
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            try:
                data[i][j] = float(col)
            except:
                continue
    return data

def removeEmptyRecord(data):
    return [row for row in data if row not in [[], None]]