import crocoddyl
import numpy as np

class ActionModelCartpoleWall(crocoddyl.ActionModelAbstract):
    def __init__(self):
        # state: 4; controls: 7 (u1,u2,u3)x(z1, z2, z3, z4); no built-in residual slots
        super().__init__(crocoddyl.StateVector(4), 3)

        # --- Parameters ---
        self.dt = 5e-2
        self.m_cart = 1.0
        self.m_pole = 1.0
        self.l_pole = 0.5
        self.grav = 10.0
        self.stiffness = 100.0   # κ (unused yet)
        self.damping = 10.0       # ν (unused yet)
        self.d_wall = 0.5

        # State / control bounds (not enforced yet)
        self.x_ub = np.array([self.d_wall, np.pi/10, 2.0, 2.0])
        self.x_lb = -self.x_ub.copy()
        self.u_ub = np.array([ 1.0,  np.inf,  np.inf])  # u1, u2, u3
        self.u_lb = np.array([-1.0, -np.inf, -np.inf])

        # Tracking reference
        self.x_ref = np.zeros(4)

        # --- Discrete-time linear dynamics x_{k+1} = A x_k + B u_k ---
        mc, mp, l, h, g = self.m_cart, self.m_pole, self.l_pole, self.dt, self.grav
        self.A = np.array([
            [1.0, 0.0, h,   0.0],
            [0.0, 1.0, 0.0, h  ],
            [0.0, (mp*g/mc)*h, 1.0, 0.0],
            [0.0, ((mc+mp)*g/(mc*l))*h, 0.0, 1.0],
        ], dtype=float)

        # last row: [ dt/(mc*l),  -dt/(mp*l),  +dt/(mp*l) ]
        self.B = np.array([
            [0.0,           0.0,           0.0],
            [0.0,           0.0,           0.0],
            [h/mc,          0.0,           0.0],
            [h/(mc*l),  -h/(mp*l),    h/(mp*l)],
        ], dtype=float)

        # LQR weights (tune as needed)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0])
        self.R = np.diag([1.0, 0.0, 0.0])

    def calc(self, data, x, u=None):
        # residual for state tracking
        data.r = x - self.x_ref

        if u is None:
            # free dynamics (no control) -- rarely used in shooting, but keep defined
            data.xnext = self.A @ x
            data.cost  = 0.5 * (data.r @ (self.Q @ data.r))
        else:
            u_c = u[:3]  # Crocoddyl will pass size nu
            data.xnext = self.A @ x + self.B @ u_c
            data.cost  = 0.5 * (data.r @ (self.Q @ data.r)) + 0.5 * (u_c @ (self.R @ u_c))

    def calcDiff(self, data, x, u=None):
        # first derivatives wrt x,u for SQP/DDP linearization
        data.Fx = self.A
        data.Lx = self.Q @ (x - self.x_ref)
        data.Lxx = self.Q

        if u is None:
            data.Fu  = np.zeros((4, self.nu))
            data.Lu  = np.zeros(self.nu)
            data.Luu = self.R
        else:
            u_c = u[:3]  # Crocoddyl will pass size nu

            data.Fu  = self.B
            data.Lu  = self.R @ u_c
            data.Luu = self.R

    def createData(self):
        return ActionDataCartpoleWall(self)

class ActionDataCartpoleWall(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        super().__init__(model)
