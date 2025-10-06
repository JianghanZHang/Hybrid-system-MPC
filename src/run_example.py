import numpy as np
import matplotlib.pyplot as plt
import crocoddyl

from cartpole_wall_action_model import ActionModelCartpoleWall
from Gurobi_MIQP_solver import GurobiCartpoleWallSolver

# Time parameters
dt = 5e-2
T = 10

# Initial state
x0 = np.array([0.1, 0.0, 1.0, 0.0])

# Create running models
runningModels = []
for t in range(T):
    action_model = ActionModelCartpoleWall()
    action_model.Q = np.eye(4) * 10.
    runningModels.append(action_model)

# Terminal model
terminalModel = ActionModelCartpoleWall()
terminalModel.Q = terminalModel.Q * 10.0

# Create shooting problem
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

# Create solver
solver = GurobiCartpoleWallSolver(problem)

# Solve
max_iter = 100
print("Solving OCP with Gurobi...")
success = solver.solve(maxiter=max_iter)

if success:
    solver.print_solution_summary()
    
    # Get solution
    sol = solver.get_solution()
    time = np.arange(T + 1) * dt
    
    # Get parameters
    l_pole = runningModels[0].l_pole
    d_wall = runningModels[0].d_wall
    
    # Compute pole tip x position: x_tip = x_cart - l*theta (from paper)
    x_cart = sol['x'][:, 0]
    theta = sol['x'][:, 1]
    x_tip = x_cart - l_pole * theta  # CORRECTED FORMULA
    
    # Visualize
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # States (first 3 on left)
    state_labels = ['Cart Position [m]', 'Pole Angle [rad]', 'Cart Velocity [m/s]']
    state_limits = [0.5, np.pi/10, 3.0]
    
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(time, sol['x'][:, i], 'b-', linewidth=2, label='Trajectory')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(state_limits[i], color='r', linestyle=':', alpha=0.5, label='Bounds')
        ax.axhline(-state_limits[i], color='r', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(state_labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Pole velocity
    ax = axes[0, 1]
    ax.plot(time, sol['x'][:, 3], 'b-', linewidth=2, label='Trajectory')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(3.0, color='r', linestyle=':', alpha=0.5, label='Bounds')
    ax.axhline(-3.0, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pole Velocity [rad/s]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pole tip x position (CORRECTED)
    ax = axes[1, 1]
    ax.plot(time, x_tip, 'purple', linewidth=2, label='Pole tip x (cart - l·θ)')
    ax.axhline(d_wall, color='r', linestyle='-', linewidth=2, alpha=0.7, label='Right wall')
    ax.axhline(-d_wall, color='r', linestyle='-', linewidth=2, alpha=0.7, label='Left wall')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(time, -d_wall, d_wall, alpha=0.1, color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pole Tip X [m]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Controls
    ax = axes[2, 0]
    ax.plot(time[:-1], sol['u'][:, 0], 'r-', linewidth=2, label='u1 (cart force)')
    ax.plot(time[:-1], sol['u'][:, 1], 'g-', linewidth=2, label='u2 (left wall)')
    ax.plot(time[:-1], sol['u'][:, 2], 'm-', linewidth=2, label='u3 (right wall)')
    ax.axhline(-1.0, color='r', linestyle=':', alpha=0.5)
    ax.axhline(1.0, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Control Forces [N]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Binary indicators
    ax = axes[2, 1]
    ax.step(time[:-1], sol['z'][:, 0], 'g-', linewidth=2, 
            label='z1: Left contact', where='post')
    ax.step(time[:-1], sol['z'][:, 2], 'm-', linewidth=2, 
            label='z3: Right contact', where='post')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Contact Indicators')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cartpole_wall_solution.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved to: cartpole_wall_solution.png")
    plt.show()
    
else:
    print("\n❌ Optimization FAILED!")