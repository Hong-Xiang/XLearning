from keras.constraints import Constraint
import keras.backend as K

class MaxMinValue(Constraint):
    def __init__(self, cmin = -0.01, cmax = 0.01):
        self.cmin = cmin
        self.cmax = cmax

    def __call__(self, p):
        p = K.clip(p, self.cmin, self.cmax)
        return p
