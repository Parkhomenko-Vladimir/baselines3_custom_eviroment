import numpy as np

class State_Env:
    def __init__(self, sizex, sizey):
        self.img = np.zeros((sizex, sizey, 3))
        self.target = np.array([])
        self.posRobot = np.array([])