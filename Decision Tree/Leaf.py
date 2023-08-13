"""
Class Leaf chứa một dictionary tên là predictions 
- predictions = {tên label: số lượng label đó}
"""

import SupportFunction

class Leaf:
    def __init__(self, final_split_data) -> None:
        self.prediction = SupportFunction.countUniqueOfData(final_split_data)