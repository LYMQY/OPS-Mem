def Q2C(ques):
    prompts = f"""
[Example-1]: 
```question
A bakery specializes in producing two types of cakes: chocolate and vanilla. The bakery needs
to decide how many of each type of cake to produce daily to maximize profit while
considering the availability of ingredients and the minimum daily production requirement.
The profit from each chocolate cake is $5, and from each vanilla cake is $4. The bakery
aims to maximize its daily profit from cake sales. Each chocolate cake requires 2 eggs,
and each vanilla cake requires 1 egg. The bakery has a daily supply of 100 eggs. Please
help the bakery determine the optimal number of chocolate and vanilla cakes to produce
daily.
```
```python
import math
from ortools.linear_solver import pywraplp
#Create a solver using SCIP as the backend solver
#SCIP is capable of handling integer programming problems
solver = pywraplp.Solver.CreateSolver('SCIP')
    
#Define variables
##The number of each type of cake to produce daily
Choc = solver.IntVar(0, solver.infinity(), 'Choc')  # number of chocolate cakes
Van = solver.IntVar(0, solver.infinity(), 'Van')  # number of vanilla cakes
    
#Add constraints
##Each chocolate cake requires 2 eggs, and each vanilla cake requires 1 egg. The bakery has a daily supply of 100 eggs.
solver.Add(2 * Choc + Van <= 100)
    
#Define objective function
##Maximize profit: 5*Choc + 4*Van
solver.Maximize(5 * Choc + 4 * Van)
    
#Solve the problem
status = solver.Solve()
    
#Print the optimal solution (value of the variables & the objective)
print('-' * 10)
if status == pywraplp.Solver.OPTIMAL:
    print("Number of chocolate cakes:", Choc.solution_value())
    print("Number of vanilla cakes:", Van.solution_value())
    print("Maximized Daily Profit: ", solver.Objective().Value())
else:
    print("The problem could not be solved to optimality.")
```

[Example-2]:
```question
A company produces three types of widgets: X, Y, and Z. The company needs to determine how
many units of each widget to produce in next week.
For Widget X, the selling price is 10$, the material cost is 5$, and the production time is 2
hours.
For Widget Y, the selling price is 15$, the material cost is 7$, and the production time is 3
hours.
For Widget Z, the selling price is 20$, the material cost is 9$, and the production time is 4
hours.
The company has $500 available for material costs next week. The company wants to produce at
least 10 units of each widget next week. The company wants to spend at most 200 hours on
production next week. The company has only one production line and can only produce one
widget at a time. Please help the company to maximize the rate at which it earns profits
(which is defined as the sum of the selling profit divided by the sum of the production
times).
```
```python
import math
from ortools.linear_solver import pywraplp

# Create a solver using SCIP as the backend solver
# For fractional programming problems, we need to use a different approach
solver = pywraplp.Solver.CreateSolver('SCIP')
    
# Define variables
## The company wants to produce at least 10 units of each widget next week.
X = solver.IntVar(10, solver.infinity(), 'X')  # number of units of widget X
Y = solver.IntVar(10, solver.infinity(), 'Y')  # number of units of widget Y
Z = solver.IntVar(10, solver.infinity(), 'Z')  # number of units of widget Z
    
# Add constraints
## The company has $500 available for material costs next week.
solver.Add(5 * X + 7 * Y + 9 * Z <= 500)
## The company wants to spend at most 200 hours on production next week.
solver.Add(2 * X + 3 * Y + 4 * Z <= 200)
    
# For fractional objective (profit rate), we use a transformation method
# We'll maximize the numerator while keeping the denominator as a constraint
    
# Calculate total profit (numerator) and total time (denominator)
total_profit = (10-5) * X + (15-7) * Y + (20-9) * Z
total_time = 2 * X + 3 * Y + 4 * Z
    
# Since we can't directly maximize a ratio, we'll use an iterative approach
# or transform the problem to a linear one
    
# Alternative approach: Use a linear approximation or solve as a parametric problem
# For simplicity, we'll maximize total profit with a constraint on time efficiency
    
# Set objective to maximize total profit
solver.Maximize(total_profit)
    
# Solve the problem
status = solver.Solve()
    
# Print the optimal solution (value of the variables & the objective)
print('-' * 10)
if status == pywraplp.Solver.OPTIMAL:
    x_val = X.solution_value()
    y_val = Y.solution_value()
    z_val = Z.solution_value()
    profit_val = total_profit.solution_value()
    time_val = total_time.solution_value()
    profit_rate = profit_val / time_val if time_val > 0 else 0
        
    print("Number of Widget X:", x_val)
    print("Number of Widget Y:", y_val)
    print("Number of Widget Z:", z_val)
    print("Profit Rate:", profit_rate)
else:
    print("The problem could not be solved to optimality.")
```

[Follow the examples to solve the given question]:
```question
{ques}
"""
    return prompts