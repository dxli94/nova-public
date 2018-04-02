class HyperBox:
    def __init__(self, bounds):
        self.bounds = bounds

    def __str__(self):
        str_repr = ''
        for idx, elem in zip(range(len(self.bounds)), self.bounds):
            str_repr += 'dimension ' + str(idx) + ': ' + str(elem) + '\n'

        return str_repr
