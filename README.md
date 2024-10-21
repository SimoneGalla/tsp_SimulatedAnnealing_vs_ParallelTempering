# tsp_SimulatedAnnealing_vs_ParallelTempering
Comparison of solution of travelling salesman problem between Simulated Annealing and Parallel Tempering algorithms


# Project description
This project implements the SA and PT algorithms in order to solve the travelling salesman problem. A data analysis from simulation results will also be included. The file "PTpotential" also includes a potential well that can be modified and shifted (it can be imagined as a river between the cities that is difficult to cross). The project is thought to be the final project for my course "Computational Physics" at KTH University, Stockholm.


# Theoretical background 
## Traveling Salesman Problem (TSP)

The **Traveling Salesman Problem (TSP)** is a well-known problem in computer science and operations research. It involves a salesman who needs to visit a set of cities, each exactly once, and return to the starting city. The goal is to find the shortest possible route that minimizes the total distance traveled.

### Key Points:
- **NP-Hard Problem**: TSP is classified as NP-hard, meaning that no efficient algorithm is known to solve all instances of the problem quickly. As the number of cities increases, the number of possible routes grows factorially, making exhaustive search impractical.
- **Applications**: TSP has real-world applications in logistics (like delivery routes), circuit design, and scheduling problems, where optimizing routes and minimizing costs is essential.

## Simulated Annealing Algorithm

**Simulated Annealing** is an optimization technique inspired by the physical process of annealing in metallurgy, where controlled heating and cooling of materials lead to improved structural integrity.

### How It Works:
1. **Initial Solution**: Start with an initial solution (a route for the TSP).
2. **Temperature Parameter**: Introduce a "temperature" parameter that controls the likelihood of accepting worse solutions.
3. **Random Moves**: At each step, make small random changes to the current solution (e.g., swapping the order of two cities).
4. **Acceptance Criteria**:
   - If the new solution is better (shorter route), accept it.
   - If it is worse, accept it with a probability that decreases as the algorithm progresses (related to the temperature).
5. **Cooling Schedule**: Gradually reduce the temperature to limit the acceptance of worse solutions over time, allowing the algorithm to settle into a more optimal solution.

### Benefits:
- **Escapes Local Minima**: The acceptance of worse solutions helps the algorithm avoid getting stuck in local minima, improving the chances of finding a global minimum.

## Parallel Tempering Algorithm

**Parallel Tempering** is a sophisticated method that improves sampling from complex probability distributions by running multiple simulations at different temperatures.

### How It Works:
1. **Multiple Replicas**: Run several independent simulations (replicas) of the TSP at different temperatures.
   - Higher temperatures allow for more exploration (greater changes in routes).
   - Lower temperatures focus on refining solutions (fine-tuning the routes).
2. **Swapping Mechanism**: Periodically, allow replicas to swap their solutions based on certain probabilities. This mechanism helps to share good solutions across different temperatures.
3. **Enhanced Exploration**: The high-temperature replicas can escape local minima more easily and share their discoveries with lower-temperature replicas, which then refine these solutions.

### Benefits:
- **Effective Exploration**: By maintaining multiple solutions at different temperatures, Parallel Tempering effectively explores the solution space, making it more likely to find optimal solutions in complex landscapes.

# Table of contents
- lin105.tsp: data for the 105 city points used to test the code algorithms
- SA.py : python file implementing the simulated annealing algorithm
- PT.py : python file implementing the Parallel Tempering algorithm
- PTpotential.py : python file implementing Parallel Tempering algorithm with a potential barrier
