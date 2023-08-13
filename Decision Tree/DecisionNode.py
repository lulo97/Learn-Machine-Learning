
class DecisionNode:
    def __init__(self, question, true_branch, false_branch) -> None:
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch