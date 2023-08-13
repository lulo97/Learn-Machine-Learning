from csv import reader

def loadDataFromCSV(csv_dir):
    with open(csv_dir, 'r') as file:
        content = reader(file)
        return list(content)