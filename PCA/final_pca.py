from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

#Các công cụ toán
mean = lambda xi_list: sum(xi_list)/len(xi_list)
std = lambda xi_list: (sum([(xi - mean(xi_list))**2 for xi in xi_list])/(len(xi_list) - 1))**0.5
standardize = lambda xi_list: [(xi - mean(xi_list))/std(xi_list) for xi in xi_list]

dot = lambda listA, listB: sum([ai*bi for ai, bi in zip(listA, listB)])
matrix_dot = lambda matrixA, matrixB: [[dot(colA, colB) for colB in zip(*matrixB)] for colA in matrixA]
transpose = lambda matrix: list(map(list, zip(*matrix)))

def covarianceMatrix(table):
    dot_result = matrix_dot(table, transpose(table))
    return [[col/(len(table[0]) - 1) for col in row] for row in dot_result]

def computeEig(cov):
    return list(np.linalg.eig(np.array(cov, dtype=np.float64)))

class PCA():
    def __init__(self, feature_data, target_data) -> None:
        self.target_data = target_data
        self.standardized = [standardize(row) for row in transpose(feature_data)]
        self.covariance_matrix = covarianceMatrix(self.standardized)
        self.eig_value, self.eig_vector = computeEig(self.covariance_matrix)

    def sortEig(self):
        self.eig_vector = transpose(self.eig_vector)
        self.eig_vector = [list(x) for _, x in sorted(zip(self.eig_value, self.eig_vector), reverse=True)]
        self.eig_vector = transpose(self.eig_vector)
        self.eig_value = sorted(self.eig_value, reverse=True)
    
    def computePCAScore(self):
        self.PCASCore = transpose(matrix_dot(transpose(self.standardized), self.eig_vector))
        self.eig_value_percent = [ele/sum(self.eig_value) for ele in self.eig_value]

    def plotPCA(self):
        PC1, PC2 = self.PCASCore[0], self.PCASCore[1]
        scatter = plt.scatter(PC1, PC2, c = self.target_data)
        plt.legend(handles=scatter.legend_elements()[0], labels=list(set(self.target_data)), loc="upper right")
        plt.xlabel(f'PC1 ({self.eig_value_percent[0]})', fontsize=15)
        plt.ylabel(f'PC2 ({self.eig_value_percent[1]})', fontsize=15)
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    #Load 1/5 dữ liệu của bộ dữ liệu breast_cancer.csv
    breast = load_breast_cancer()

    breast_data = breast.data
    breast_target = breast.target

    print(len(breast.feature_names), breast.feature_names)

    #Chạy thuật toán
    pca = PCA(breast_data[0:len(breast.data)//5], breast_target[0:len(breast.data)//5])
    pca.sortEig()
    pca.computePCAScore()
    pca.plotPCA()

