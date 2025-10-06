import gurobipy as gp
from gurobipy import GRB
import numpy as np
import crocoddyl

class GurobiCartpoleWallSolver:
    """
    Gurobi-based MIQP solver for cart-pole with wall contacts using indicator constraints.
    Uses MVar for efficient vectorized constraints.
    """
    
    def __init__(self, problem):
        """
        Args:
            problem: crocoddyl.ShootingProblem instance
        """
        self.problem = problem
        self.T = len(problem.runningModels)
        self.nx = problem.x0.shape[0]
        
        # Extract parameters from first running model
        model = problem.runningModels[0]
        self.dt = model.dt
        self.mc = model.m_cart
        self.mp = model.m_pole
        self.l = model.l_pole
        self.g = model.grav
        self.kappa = model.stiffness
        self.nu = model.damping
        self.d = model.d_wall
        
        # State and control bounds
        self.x_lb = model.x_lb
        self.x_ub = model.x_ub
        self.u1_lb, self.u1_ub = model.u_lb[0], model.u_ub[0]
        
        self.nu_continuous = 3  # u1, u2, u3
        self.nu_binary = 4  # z1, z2, z3, z4
        
        # Solution storage
        self.xs = [np.zeros(self.nx) for _ in range(self.T + 1)]
        self.us = [np.zeros(self.nu_continuous) for _ in range(self.T)]
        self.zs = [np.zeros(self.nu_binary) for _ in range(self.T)]
        
        # Gurobi model
        self.gurobi_model = None
        self.vars = {}
        self.cost = 0.0
        self.solve_time = 0.0
        self.mip_gap = 0.0
        
    def solve(self, xs_init=None, us_init=None, maxiter=100, isFeasible=False):
        """
        Solve the OCP.
        
        Args:
            xs_init: list of initial state guesses (T+1 elements)
            us_init: list of initial control guesses (T elements)
            maxiter: maximum number of iterations (used as time limit in seconds)
            isFeasible: whether the initial guess is feasible (not used for MIQP)
        
        Returns:
            success: True if optimal solution found
        """
        # Build the optimization problem
        self._build_problem()
        
        # Set warm start if provided
        if xs_init is not None and us_init is not None:
            try:
                self._set_warm_start(xs_init, us_init)
            except Exception as e:
                print(f"Warning: Could not set warm start: {e}")
        
        # Set time limit
        self.gurobi_model.setParam('TimeLimit', maxiter)
        
        # Solve
        self.gurobi_model.optimize()
        
        # Check status
        status = self.gurobi_model.status
        if status == GRB.OPTIMAL:
            self._extract_solution()
            return True
        elif status == GRB.TIME_LIMIT:
            if self.gurobi_model.SolCount > 0:
                print("Time limit reached. Extracting best solution found...")
                self._extract_solution()
                return True
            return False
        elif status == GRB.INFEASIBLE:
            print("\n" + "="*60)
            print("MODEL IS INFEASIBLE!")
            print("="*60)
            print("Computing IIS (Irreducible Inconsistent Subsystem)...")
            self.gurobi_model.computeIIS()
            self.gurobi_model.write("model_infeasible.ilp")
            print("IIS written to model_infeasible.ilp")
            
            # Print conflicting constraints
            print("\nConflicting constraints:")
            for c in self.gurobi_model.getConstrs():
                if c.IISConstr:
                    print(f"  - {c.ConstrName}")
            
            print("\nConflicting bounds:")
            for v in self.gurobi_model.getVars():
                if v.IISLB:
                    print(f"  - {v.VarName} >= {v.LB}")
                if v.IISUB:
                    print(f"  - {v.VarName} <= {v.UB}")
            print("="*60)
            return False
        else:
            print(f"Optimization failed with status {status}")
            return False
    
    def _build_problem(self):
        """Build the Gurobi optimization problem using MVar."""
        m = gp.Model("CartpoleWall_MIQP")
        m.setParam('OutputFlag', 1)
        m.setParam('MIPGap', 1e-4)
        
        # States: x[t] for t = 0..T (shape: T+1 x nx)
        x = m.addMVar((self.T + 1, self.nx), lb=self.x_lb, ub=self.x_ub, name="x")
        
        # Continuous controls: u1, u2, u3 for t = 0..T-1
        u1 = m.addMVar(self.T, lb=self.u1_lb, ub=self.u1_ub, name="u1")
        u2 = m.addMVar(self.T, lb=0, name="u2")
        u3 = m.addMVar(self.T, lb=0, name="u3")
        
        # Binary indicators
        z1 = {}
        z2 = {}
        z3 = {}
        z4 = {}
        z_both_left = {}
        z_both_right = {}
        
        for t in range(self.T):
            z1[t] = m.addVar(vtype=GRB.BINARY, name=f"z1_{t}")
            z2[t] = m.addVar(vtype=GRB.BINARY, name=f"z2_{t}")
            z3[t] = m.addVar(vtype=GRB.BINARY, name=f"z3_{t}")
            z4[t] = m.addVar(vtype=GRB.BINARY, name=f"z4_{t}")
            z_both_left[t] = m.addVar(vtype=GRB.BINARY, name=f"z_both_left_{t}")
            z_both_right[t] = m.addVar(vtype=GRB.BINARY, name=f"z_both_right_{t}")
        
        # Auxiliary scalar variables
        delta_left = m.addMVar(self.T, lb=-GRB.INFINITY, name="delta_left")
        delta_right = m.addMVar(self.T, lb=-GRB.INFINITY, name="delta_right")
        delta_dot_left = m.addMVar(self.T, lb=-GRB.INFINITY, name="delta_dot_left")
        delta_dot_right = m.addMVar(self.T, lb=-GRB.INFINITY, name="delta_dot_right")
        force_expr_left = m.addMVar(self.T, lb=-GRB.INFINITY, name="force_expr_left")
        force_expr_right = m.addMVar(self.T, lb=-GRB.INFINITY, name="force_expr_right")
        
        m.update()
        
        # Initial state constraint
        x0 = self.problem.x0
        m.addConstr(x[0] == x0, name="init_state")
        
        # Pole tip is at x1 - l*x2
        # Left wall at -d, right wall at +d
        C_delta_left = np.array([-1.0, self.l, 0.0, 0.0])   # delta_left = -x1 + l*x2 - d
        C_delta_right = np.array([1.0, -self.l, 0.0, 0.0])  # delta_right = x1 - l*x2 - d
        C_delta_dot_left = np.array([0.0, 0.0, -1.0, self.l])
        C_delta_dot_right = np.array([0.0, 0.0, 1.0, -self.l])
        
        # Constraints for each time step
        for t in range(self.T):
            model_t = self.problem.runningModels[t]
            A_t = model_t.A
            B_t = model_t.B
            
            # ======== Auxiliary variables========
            # Penetrations
            m.addConstr(delta_left[t] == C_delta_left @ x[t] - self.d, 
                       name=f"delta_left_def_{t}")
            m.addConstr(delta_right[t] == C_delta_right @ x[t] - self.d, 
                       name=f"delta_right_def_{t}")
            m.addConstr(delta_dot_left[t] == C_delta_dot_left @ x[t], 
                       name=f"delta_dot_left_def_{t}")
            m.addConstr(delta_dot_right[t] == C_delta_dot_right @ x[t], 
                       name=f"delta_dot_right_def_{t}")
            
            # Forces
            m.addConstr(force_expr_left[t] == self.kappa * delta_left[t] + self.nu * delta_dot_left[t], 
                       name=f"force_left_expr_{t}")
            m.addConstr(force_expr_right[t] == self.kappa * delta_right[t] + self.nu * delta_dot_right[t], 
                       name=f"force_right_expr_{t}")
            
            # ========================================

            # ================ Binary variables (logical constraints)================
            # LEFT WALL INDICATORS
            m.addGenConstrIndicator(z1[t], True, delta_left[t] >= 0, 
                                   name=f"left_contact_on_{t}")
            m.addGenConstrIndicator(z1[t], False, delta_left[t] <= 0, 
                                   name=f"left_contact_off_{t}")
            m.addGenConstrIndicator(z2[t], True, force_expr_left[t] >= 0, 
                                   name=f"left_force_feasible_on_{t}")
            m.addGenConstrIndicator(z2[t], False, force_expr_left[t] <= 0, 
                                   name=f"left_force_feasible_off_{t}")
            m.addGenConstrIndicator(z1[t], False, u2[t] == 0, 
                                   name=f"left_no_contact_{t}")
            m.addGenConstrIndicator(z2[t], False, u2[t] == 0, 
                                   name=f"left_force_infeasible_{t}")
            
            m.addConstr(z_both_left[t] <= z1[t], name=f"both_left_1_{t}")
            m.addConstr(z_both_left[t] <= z2[t], name=f"both_left_2_{t}")
            m.addConstr(z_both_left[t] >= z1[t] + z2[t] - 1, name=f"both_left_3_{t}")
            m.addGenConstrIndicator(z_both_left[t], True, u2[t] == force_expr_left[t], 
                                   name=f"left_force_active_{t}")
            
            # RIGHT WALL INDICATORS
            m.addGenConstrIndicator(z3[t], True, delta_right[t] >= 0, 
                                   name=f"right_contact_on_{t}")
            m.addGenConstrIndicator(z3[t], False, delta_right[t] <= 0, 
                                   name=f"right_contact_off_{t}")
            m.addGenConstrIndicator(z4[t], True, force_expr_right[t] >= 0, 
                                   name=f"right_force_feasible_on_{t}")
            m.addGenConstrIndicator(z4[t], False, force_expr_right[t] <= 0, 
                                   name=f"right_force_feasible_off_{t}")
            m.addGenConstrIndicator(z3[t], False, u3[t] == 0, 
                                   name=f"right_no_contact_{t}")
            m.addGenConstrIndicator(z4[t], False, u3[t] == 0, 
                                   name=f"right_force_infeasible_{t}")
            
            m.addConstr(z_both_right[t] <= z3[t], name=f"both_right_1_{t}")
            m.addConstr(z_both_right[t] <= z4[t], name=f"both_right_2_{t}")
            m.addConstr(z_both_right[t] >= z3[t] + z4[t] - 1, name=f"both_right_3_{t}")
            m.addGenConstrIndicator(z_both_right[t], True, u3[t] == force_expr_right[t], 
                                   name=f"right_force_active_{t}")
            # ========================================

            # ========== Continuous variables ================
            # DYNAMICS
            u_t = gp.hstack([u1[t], u2[t], u3[t]])
            m.addConstr(x[t+1] == A_t @ x[t] + B_t @ u_t, name=f"dyn_{t}")
        
        # OBJECTIVE
        obj = 0.0
        
        for t in range(self.T):
            model_t = self.problem.runningModels[t]
            Q_t = model_t.Q
            R_t = model_t.R
            x_ref_t = model_t.x_ref
            
            x_err = x[t] - x_ref_t
            obj += 0.5 * x_err @ Q_t @ x_err
            
            u_t = gp.hstack([u1[t], u2[t], u3[t]])
            obj += 0.5 * u_t @ R_t @ u_t
        
        model_T = self.problem.terminalModel
        Q_T = model_T.Q
        x_ref_T = model_T.x_ref
        x_T_err = x[self.T] - x_ref_T
        obj += 0.5 * x_T_err @ Q_T @ x_T_err
        
        m.setObjective(obj, GRB.MINIMIZE)
        
        self.gurobi_model = m
        self.vars = {
            'x': x, 'u1': u1, 'u2': u2, 'u3': u3,
            'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4,
            'delta_left': delta_left, 'delta_right': delta_right
        }
    
    def _set_warm_start(self, xs_init, us_init):
        """Set warm start values."""
        x = self.vars['x']
        u1 = self.vars['u1']
        u2 = self.vars['u2']
        u3 = self.vars['u3']
        
        for t in range(min(self.T + 1, len(xs_init))):
            x[t].start = xs_init[t]
        
        for t in range(min(self.T, len(us_init))):
            u1[t].start = us_init[t][0]
            if len(us_init[t]) > 1:
                u2[t].start = us_init[t][1]
                u3[t].start = us_init[t][2]
        
    def _extract_solution(self):
        """Extract solution from Gurobi model."""
        x = self.vars['x']
        u1 = self.vars['u1']
        u2 = self.vars['u2']
        u3 = self.vars['u3']
        delta_left = self.vars['delta_left']
        delta_right = self.vars['delta_right']
        
        self.deltas_left = np.zeros(self.T)
        self.deltas_right = np.zeros(self.T)
        
        for t in range(self.T + 1):
            self.xs[t] = x[t].X
            
            if t < self.T:
                self.us[t] = np.array([u1[t].X, u2[t].X, u3[t].X])
                self.zs[t] = np.array([
                    self.vars['z1'][t].X,
                    self.vars['z2'][t].X,
                    self.vars['z3'][t].X,
                    self.vars['z4'][t].X
                ])
                self.deltas_left[t] = delta_left[t].X
                self.deltas_right[t] = delta_right[t].X
        
        self.cost = self.gurobi_model.ObjVal
        self.solve_time = self.gurobi_model.Runtime
        self.mip_gap = self.gurobi_model.MIPGap
    
    def get_solution(self):
        """Return solution as dict."""
        return {
            'x': np.array(self.xs),
            'u': np.array(self.us),
            'z': np.array(self.zs),
            'delta_left': self.deltas_left,
            'delta_right': self.deltas_right,
            'cost': self.cost,
            'solve_time': self.solve_time,
            'mip_gap': self.mip_gap
        }
    
    def print_solution_summary(self):
        """Print summary of the solution."""
        print("\n" + "="*60)
        print("GUROBI SOLVER SUMMARY")
        print("="*60)
        print(f"Total Cost:      {self.cost:.6f}")
        print(f"Solve Time:      {self.solve_time:.3f} seconds")
        print(f"MIP Gap:         {self.mip_gap:.6f}")
        print(f"\nInitial State:   {self.xs[0]}")
        print(f"Final State:     {self.xs[-1]}")
        print(f"\nMax |u1|:        {np.max(np.abs([u[0] for u in self.us])):.4f}")
        print(f"Max u2 (left):   {np.max([u[1] for u in self.us]):.4f}")
        print(f"Max u3 (right):  {np.max([u[2] for u in self.us]):.4f}")
        print(f"\nLeft contacts:   {np.sum([z[0] for z in self.zs])} / {self.T}")
        print(f"Right contacts:  {np.sum([z[2] for z in self.zs])} / {self.T}")
        print("="*60 + "\n")