import math

class Activation:

    @staticmethod
    def sigmoid(x: float):
        if abs(x) > 700:
            return 1 if x > 0 else 0
        return 1/(1 + math.e**(-x))

    @staticmethod
    def derivativeSigmoid(x: float):
        return x * (1 - x)

    @staticmethod
    def computeVector(v: list):
        return [Activation.sigmoid(vi) for vi in v] 
        
    @staticmethod
    def computeMatrix(m: list[list]):
        return [Activation.computeVector(row) for row in m]
        
    @staticmethod
    def derivativeVector(v: list):
        return [Activation.derivativeSigmoid(vi) for vi in v] 
        
    @staticmethod
    def derivativeMatrix(m: list[list] ):
        return [Activation.derivativeVector(row) for row in m]
        
if __name__ == "__main__":
    x = 1
    v = [1, 2, 3]
    m = [[1, 2, 3], [4, 5, 6]]

    func_name = "relu"

    print(Activation.compute(x, func_name))
    print(Activation.computeVector(v, func_name))
    print(Activation.computeMatrix(m, func_name))

    print(Activation.derivative(x, func_name))
    print(Activation.derivativeVector(v, func_name))
    print(Activation.derivativeMatrix(m, func_name))