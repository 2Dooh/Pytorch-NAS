from pymoo.model.duplicate import ElementwiseDuplicateElimination

import numpy as np

import logging

class DARTSDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, **kwargs) -> None:
        super().__init__(cmp_func=self.is_equal, **kwargs)

    def is_equal(self, a, b):
        x = a.get('X'); y = b.get('X')

        result = self.compare(x.tolist(), y.tolist())
        if result:
            logging.info('dupplicate: - {} = {}'.format(x, y))
        return result

    @staticmethod
    def convert_cell(cell_bit_string):
        # convert cell bit-string to genome
        tmp = [cell_bit_string[i:i + 2] for i in range(0, len(cell_bit_string), 2)]
        return [tmp[i:i + 2] for i in range(0, len(tmp), 2)]

    def compare_cell(self, cell_string1, cell_string2):
        cell_genome1 = self.convert_cell(cell_string1)
        cell_genome2 = self.convert_cell(cell_string2)
        cell1, cell2 = cell_genome1[:], cell_genome2[:]

        for block1 in cell1:
            for block2 in cell2:
                if block1 == block2 or block1 == block2[::-1]:
                    cell2.remove(block2)
                    break
        if len(cell2) > 0:
            return False
        else:
            return True

    def compare(self, string1, string2):

        if self.compare_cell(string1[:len(string1)//2],
                        string2[:len(string2)//2]):
            if self.compare_cell(string1[len(string1)//2:],
                            string2[len(string2)//2:]):
                return True

        return False