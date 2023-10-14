if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Model
from dezero.utils import plot_dot_graph
from dezero.utils import sum_to
import dezero.functions as F
import matplotlib.pyplot as plt
import dezero.layers as L
from dezero.models import MLP
from dezero import optimizers
from dezero import as_variable
import dezero
import math


class MyIterator:
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()
        
        self.cnt += 1
        return self.cnt
    

obj = MyIterator(5)
for x in obj:
    print(x)