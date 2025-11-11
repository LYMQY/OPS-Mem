from ortools.linear_solver import pywraplp

class NurseSchedulingProblem:
    def __init__(self):
        # Sets
        self.T = [1, 2, 3, 4, 5, 6]  # Periods
        self.S = [2, 6, 10, 14, 18, 22]  # Shift start times

        # Demand parameters
        self.D = {
            1: 10,
            2: 15,
            3: 25,
            4: 20,
            5: 18,
            6: 12
        }

        # Cost parameters
        self.c_r = 10  # Regular staff cost per hour
        self.c_o = 15  # Overtime cost per hour
        self.c_c = 15  # Contract nurse cost per hour

        # Workforce constraints
        self.R_max = 40    # Max regular staff nurses
        self.O_max = 10    # Max overtime nurses
        self.C_min = 2     # Min contract nurses per period
        self.alpha = 0.6   # Minimum regular staff coverage ratio

        # Create the solver
        self.solver = pywraplp.Solver.CreateSolver('SCIP')

        # Initialize variables
        self.x4 = {}  # Regular staff nurses on 4-hour shifts
        self.x8 = {}  # Regular staff nurses on 8-hour shifts
        self.y = {}   # Overtime nurses (12-hour shifts)
        self.z4 = {}  # Contract nurses on 4-hour shifts
        self.z8 = {}  # Contract nurses on 8-hour shifts

    def define_variables(self):
        # Define decision variables
        for s in self.S:
            self.x4[s] = self.solver.IntVar(0, self.solver.infinity(), f'x4_{s}')
            self.x8[s] = self.solver.IntVar(0, self.solver.infinity(), f'x8_{s}')
            self.y[s] = self.solver.IntVar(0, self.solver.infinity(), f'y_{s}')
            self.z4[s] = self.solver.IntVar(0, self.solver.infinity(), f'z4_{s}')
            self.z8[s] = self.solver.IntVar(0, self.solver.infinity(), f'z8_{s}')

    def define_constraints(self):
        # Constraint 2: Regular staff cap
        self.solver.Add(sum(self.x4[s] + self.x8[s] for s in self.S) <= self.R_max)

        # Constraint 3: Overtime utilization limit
        self.solver.Add(sum(self.y[s] for s in self.S) <= self.O_max)

        # Determine which shifts are active in each period
        # This is predefined based on the time periods and shifts
        # Format: {period: {shift_length: [shift_start_times_active_in_this_period]}}
        shift_activity = {
            1: {4: [2], 8: [2, 6], 12: [2]},
            2: {4: [6], 8: [2, 6, 10], 12: [2]},
            3: {4: [10], 8: [6, 10, 14], 12: [2, 6]},
            4: {4: [14], 8: [10, 14, 18], 12: [6, 10]},
            5: {4: [18], 8: [14, 18, 22], 12: [10, 14]},
            6: {4: [22], 8: [18, 22], 12: [14, 18]}
        }

        for t in self.T:
            # Constraint 1: Shift coverage
            active_shifts = shift_activity[t]
            coverage = self.solver.Sum([
                *[self.x4[s] + self.z4[s] for s in active_shifts.get(4, [])],
                *[self.x8[s] + self.z8[s] for s in active_shifts.get(8, [])],
                *[self.y[s] for s in active_shifts.get(12, [])]
            ])
            self.solver.Add(coverage >= self.D[t])

            # Constraint 4: Minimum contract nurses
            contract_nurses = self.solver.Sum([
                *[self.z4[s] for s in active_shifts.get(4, [])],
                *[self.z8[s] for s in active_shifts.get(8, [])]
            ])
            self.solver.Add(contract_nurses >= self.C_min)

            # Constraint 5: Minimum regular coverage
            regular_coverage = self.solver.Sum([
                *[self.x4[s] for s in active_shifts.get(4, [])],
                *[self.x8[s] for s in active_shifts.get(8, [])],
                *[self.y[s] for s in active_shifts.get(12, [])]
            ])
            self.solver.Add(regular_coverage >= self.alpha * self.D[t])

        # Constraint 6: Logical constraint (overtime <= regular 8-hour shifts)
        for s in self.S:
            self.solver.Add(self.y[s] <= self.x8[s])

    def define_objective(self):
        # Objective: Minimize total daily cost
        cost = self.solver.Sum([
            4 * self.c_r * (self.x4[s] + 2 * self.x8[s]) +
            4 * self.c_o * self.y[s] +
            4 * self.c_c * (self.z4[s] + 2 * self.z8[s])
            for s in self.S
        ])
        self.solver.Minimize(cost)

    def solve(self):
        # Define variables, constraints, and objective
        self.define_variables()
        self.define_constraints()
        self.define_objective()

        # Solve the problem
        status = self.solver.Solve()

        # Print results
        if status == pywraplp.Solver.OPTIMAL:
            print(f'Optimal solution found with total cost: ${int(self.solver.Objective().Value())}')
            print('\nRegular Staff Nurses:')
            print(f"{'Shift Start':<12} {'4-hour':<8} {'8-hour':<8} {'Overtime':<8}")
            for s in self.S:
                print(f"{s:<12} {int(self.x4[s].solution_value()):<8} {int(self.x8[s].solution_value()):<8} {int(self.y[s].solution_value()):<8}")

            print('\nContract Nurses:')
            print(f"{'Shift Start':<12} {'4-hour':<8} {'8-hour':<8}")
            for s in self.S:
                print(f"{s:<12} {int(self.z4[s].solution_value()):<8} {int(self.z8[s].solution_value()):<8}")
        else:
            print('No optimal solution found.')

def main():
    nsp = NurseSchedulingProblem()
    nsp.solve()

if __name__ == '__main__':
    main()