from random import randint
from random import uniform
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from numpy.random import rand as rnd

def read_tsp_coordinates(file_path):
    """
    Reads a TSP file and extracts the list of coordinates.

    Args:
        file_path (str): Path to the TSP file.

    Returns:
        list: A list of tuples, where each tuple represents (x, y) coordinates.
    """
    coordinates = []
    with open(file_path, 'r') as file:
        # Skip lines until we reach the NODE_COORD_SECTION
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                break

        # Read the coordinates section
        for line in file:
            line = line.strip()
            if line == "EOF":  # End of the coordinates section
                break
            parts = line.split()
            # Extract the index, x, and y values
            index, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            coordinates.append((x, y))

    return coordinates

def distance(start, arrive):
    """
    Calculates the distance between two cities.

    Parameters:
    ----------
        start: tuple of float
            Starting point (latitude, longitude).
        arrive : tuple of float
            Arriving point (latitude, longitude).

    Returns:
    -------
    float
        Distance between the two cities.
    """

    d = np.sqrt((start[0] - arrive[0]) ** 2 + (start[1] - arrive[1]) ** 2)
    return d


def Fitness(cities):
    """Calculates the total distances of a path between all the points in an ordered list of points.
    The last step goes from the last point in the list to the first.
    Parameters:
            cities (list) : list of points needed to travel, in random order
        Returns:
            totDist: total distance between the cities in the given order"""

    # Initialize the distance in order to  sum progressively all the distances
    TotDistance = 0

    # Now I will iterate over the cities and calculate the total fitness
    for i in range(len(cities)):
        startCity = cities[i]
        endCity = None

        if i + 1 < len(cities):
            endCity = cities[i + 1]
        else:
            endCity = cities[0]
        TotDistance += distance(startCity, endCity)
    return TotDistance


def swap(cities):
    """Swap cities at positions i and j with each other
    Parameters:
        cities (list) : ordered list of the cities
    Returns:
        The neighbour list of cities
        """
    pos_one = randint(0, len(cities)-1)
    pos_two = randint(0, len(cities)-1)
    cities[pos_one], cities[pos_two] = cities[pos_two], cities[pos_one]

    return cities

def inverse(cities):
    """Inverses the order of the values between two cities at random
    Parameters:
        cities (list) : ordered list of the cities
    Returns:
        The neighbour list of cities
        """

    i = randint(0, len(cities)-1)
    firstChoice = cities[i]

    newCities = list(filter(lambda city: city != firstChoice, cities))
    j = randint(0, len(newCities)-1)
    secondChoice = newCities[j]

    # Now let's reverse all the values between first and second choice
    cities[min(cities.index(firstChoice), cities.index(secondChoice)):max(cities.index(firstChoice), cities.index(secondChoice))] = cities[min(cities.index(firstChoice), cities.index(secondChoice)):max(
        cities.index(firstChoice), cities.index(secondChoice))][::-1]

    return cities

def swap_routes(state):
    """Select a subroute from a to b and insert it at another position in the route
    Parameters:
        cities (list) : ordered list of the cities
    Returns:
        The neighbour list of cities
        """
    subroute_a = randint(0, len(state)-1)
    subroute_b = randint(0, len(state)-1)
    subroute = state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
    del state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
    insert_pos = randint(0, len(state) -1)
    for i in subroute:
        state.insert(insert_pos, i)
    return state

def two_opt(cities):
    """Takes a random segment and inverses it within the list
    Parameters:
        cities (list) : ordered list of the cities
    Returns:
        The neighbour list of cities
        """

    i, j = sorted(random.sample(range(len(cities)), 2))
    cities[i:j] = cities[i:j][::-1]  # Reverse the segment
    return cities


def or_opt(cities):
    """Remove a small segment of cities and reinsert it at a new position.
    Parameters:
        cities (list) : ordered list of the cities
    Returns:
        The neighbour list of cities
        """
    n = len(cities)

    # Select random start point for the subsegment
    start = randint(0, n - 2)

    # Randomly decide the length of the subsegment (1, 2, or 3 cities)
    segment_length = randint(1, min(3, n - start))

    subsegment = cities[start:start + segment_length]

    remaining_cities = cities[:start] + cities[start + segment_length:]

    insertion_point = randint(0, len(remaining_cities))

    # Insert the segment at the new position
    new_route = remaining_cities[:insertion_point] + subsegment + remaining_cities[insertion_point:]

    return new_route

def plotta(cities, colore ="y", title = "Route", show = True):
    """
        Plots the cities and the connection between them in the order given by cities

        Parameters:
            cities (list) : list of the city in a certain order
            colore (string) : color of the lines connecting the cities
            title (string) : title of the plot
            show (bool) : shows the plot (might be needed to not show)

        Returns:
            A plot of the cities connected to each other in the given order    """


    # Take the values from the city list route
    xvalues = [city[0] for city in cities]
    yvalues = [city[1] for city in cities]
    xvalues.append(cities[0][0])
    yvalues.append(cities[0][1])

    # Plots the cities and the connections following the given order
    if show:
        fig, ax = plt.subplots()
        ax.plot(xvalues, yvalues, '--', color=colore)
        ax.set_xlabel('X Coordinates')
        ax.set_ylabel('Y Coordinates')
        plt.title(title)

        for city in cities:
            ax.scatter(city[0], city[1], color='red')
        plt.show()

    # Might need only the ordered values without the plot (Not used here)
    if not show:
        return xvalues, yvalues
def annealing(initial, Nmax = 10000):
    """
    This function implements the simulated annealing algorithm. In this algorithm 
    Parameters:
        initial (list) : input list of cities
    Returns:

         solution (list) : output list of cities after simulated annealing algorithm

         totalDistance (float) : total distance of the path between the cities in the order given by solution
    """

    # Set the initial parameters

    initialTemp = 1
    alpha = 0.99  # Will slowly lower the temperature
    T = initialTemp
    solution = initial

    sameDistance = 0
    sameSolution = 0
    N = 0

    # Now let's perform the simulation steps for Nmax times
    while N < Nmax:

        # Randomly choose a way of selecting a neighbour and select it

        number = randint(1,2)
        if number == 1:
            neighbour = or_opt(solution[:])
        elif number ==2:
            neighbour = inverse(solution[:])
        elif number == 3:
            neighbour = swap_routes(solution[:])
        elif number == 4:
            neighbour = two_opt(solution[:])

        # Find the difference in distance between the previous solution and the neighbour
        distanceDiff = - Fitness(neighbour) + Fitness(solution[:])

        if distanceDiff > 0:
            solution = neighbour
            sameDistance = 0
            sameSolution = 0

        elif distanceDiff == 0:
            sameDistance += 1
            sameSolution += 1

        # Now let's see if the solution is not better

        elif distanceDiff < 0:
            if random.gauss(0.5,0.005) <= np.exp((distanceDiff) / T): #Accept the change with a certain probability
                solution = neighbour
                sameDistance = 0
                sameSolution = 0
            else:
                sameDistance += 1
                sameSolution += 1

        T = T*alpha # Update the temperature and the number of iterations
        N+=1

    print("The final result is:", Fitness(solution))

    return solution, Fitness(solution)


def distanceMatrix(cities):

    #This is needed to implement the Nearest Neighbour Search algorithm

    grid = cities
    distance_grid = np.empty((len(grid), len(grid)))

    for i in range(len(grid)):
        for j in range(len(grid)):
            distance_grid[i, j] = distance(grid[i], grid[j])

    return distance_grid

def NNS(cities):
    """This function starts from a city and finds its nearest neighbour and goes on from that city to the next one.
    The process keeps going, and it will return a result that is not optimal, but it is a good starting point for the
    Simulated Annealing algorithm that will follow
    Parameters:
          cities (list) : ordered initial list of cities points
    Returns:
          result (list) : ordered list of city points where each city is followed by the its closest neighbour thanks to the NNS algorithm"""

    free_nodes = []

    for i in range(len(cities) - 1):
        free_nodes.append(i + 1)
    current_node = 0
    solution = [current_node]

    while free_nodes:
        next_node = min(
            free_nodes, key=lambda x: distance(cities[current_node], cities[x]))
        free_nodes.remove(next_node)
        solution.append(next_node)
        current_node = next_node

    i = 0
    result = list(np.zeros(len(cities)))
    for ind in solution:
        result[i] = cities[ind]
        i+=1

    print("Best Distance after NNS is:", distance)

    return result



choice = input("Random cities o from a database?") #This will give either a random set of cities or the 105 cities given in the file

if choice == "Random" or choice == "random":
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(100)]


else: # This will return the standard set of 105 cities that has a known solution
    path = "lin105.tsp"
    cities = read_tsp_coordinates(path)


plotta(cities, colore = "violet", title = "Random initial route")  # Plots the initial random route
routes = []
distances = []

# Perform the Nearest Neighbour Search algorithm to find an initial route

start = time.time()
bestRoute = NNS(cities)
end = time.time()

print("Time for NNS is:", end - start)
plotta(bestRoute, title = "Route after NNS")


initDistance = Fitness(bestRoute)
print("The initial distance after NNS is: ", initDistance)

# Now let's perform the Simulated Annealing algorithm keeping track of the time
start = time.time()

Nmax = 100 #Number of iterations of the annealing algorithm

for _ in range(90): # The simulation is resetted and performed N times in order to have different trials and to not get stuck in some minima

    bestRoute, bestRouteDistance = annealing(bestRoute, Nmax) # Do the annealing starting with the best found route at the earlier iteration

    # Store the route and distance for analysis later
    routes.append(bestRoute[:])
    distances.append(bestRouteDistance)


end = time.time()
timeConvergence = end - start
print("The necessary time is:", timeConvergence)


# Find the best overall route and plot it at the end
i = distances.index(min(distances))
j = 0
for element in routes:
    if j == i:
        Route = element

    j+=1

print(Fitness(Route))

distance = distances[i]

print("Best Distance is:", distance)

plotta(bestRoute, colore = "green", title = "Best found route") # Plots the best found route
