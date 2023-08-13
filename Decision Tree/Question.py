"""
Class Question có:
int column là chỉ số thuộc tính cần xét
int, str value là giá trị threshold
list header là tập các loại của y 

def match(row): trả về True nếu row này thỏa mãn Question, False nếu ngược lại

def __repr__(): in ra câu trả lời dễ đọc

*Câu hỏi sẽ nhận input là dạng số hoặc chữ
"""

def isNumber(_input):
    return isinstance(_input, int) or isinstance(_input, float)

class Question:
    def __init__(self, column: int, value: object, header: list) -> None:
        self.column = column
        self.value = value
        self.header = header
    
    def match(self, row: list):
        consider_attribute = row[self.column]
        if isNumber(consider_attribute):
            return consider_attribute >= self.value
        return consider_attribute == self.value

    def __repr__(self) -> str:
        sign = "=="
        if isNumber(self.value):
            sign = ">="
        return f"{self.header[self.column]} {sign} {self.value}?"
    
if __name__ == "__main__":
    q = Question(1, 45, ["height", "weight", "iq"])
    print(q.match([50, 76, 99]))

    q2 = Question(0, "apple", ["like", "weight", "iq"])
    print(q2.match(["apple", 76, 99]))
