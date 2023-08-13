import math

def transpose(matrix: list[list]):
    return list(map(list, zip(*matrix)))

def norm(vectorA: list, vectorB: list):
    return sum([(a - b)**2 for a, b in zip(vectorA, vectorB)])**0.5

def RMSEMatrix(matrixA: list[list], matrixB: list[list]):
    return (sum([norm(ai, bi)**2 for ai, bi in zip(matrixA, matrixB)])/len(matrixA))**0.5

class Add:
    @staticmethod
    def matrix(matrixA: list[list], matrixB: list[list]):
        return [[aij + bij for aij, bij in zip(ai, bi)] for ai, bi in zip(matrixA, matrixB)]

class Subtract:
    def __init__(self) -> None:
        pass

    @staticmethod
    def matrix(matrixA: list[list], matrixB: list[list]):
        return [[aij - bij for aij, bij in zip(ai, bi)] for ai, bi in zip(matrixA, matrixB)]
    
class Mutiply:
    @staticmethod
    def matrix(matrixA: list[list], matrixB: list[list]):
        return [[sum(a*b for a, b in zip(rowA, colB)) for colB in zip(*matrixB)] for rowA in matrixA]        

    @staticmethod
    def eleByEleVector(vectorA: list[float], vectorB: list[float]):
        return [ai*bi for ai, bi in zip(vectorA, vectorB)]

    @staticmethod
    def eleByEleMatrix(matrixA: list[list], matrixB: list[list]):
        return [[aij*bij for aij, bij in zip(ai, bi)] for ai, bi in zip(matrixA, matrixB)]

    @staticmethod
    def dotProduct(vectorA: list[float], vectorB: list[float]):
        return sum(Mutiply.eleByEle(vectorA, vectorB))

    def constToMatrix(const, matrix: list[list]):
        return [[const*cell for cell in row] for row in matrix]