import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.0
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1**2 + self.l1**2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2**2 + self.l2**2)
        self.m3 = 0.0
        self.r3 = 0.01
        self.I_3 = 0
        self.d1 = self.l1 / 2
        self.d2 = self.l2 / 2
        # self.I_3 = 2.0 / 5 * self.m3 * self.r3**2

        self.alpha = (
                self.m1 * self.d1 ** 2
                + self.I_1
                + self.m2 * (self.l1 ** 2 + self.d2 ** 2)
                + self.I_2
        )
        self.beta = self.m2 * self.l1 * self.d2
        self.gamma = self.m2 * self.d2 ** 2 + self.I_2
    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        matrix = [
            [
                self.alpha + 2 * self.beta * np.cos(q2),
                self.gamma + self.beta * np.cos(q2),
            ],
            [self.gamma + self.beta * np.cos(q2), self.gamma],
        ]
        return matrix
    
    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        return NotImplementedError()
