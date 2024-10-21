from random import randint
from random import uniform
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from numpy.random import rand as rnd

def colors(n=100):

    """Finds a list of random colors of the given length with high saturation and lightness in order to have a
    clear plot with different points. It will be used to plot the temperature swaps between the systems.

    Parameters:
          n (int) : the number of colors to generate
          colors (list) : a list of random colors of length n"""

    colors = []
    for _ in range(n):
        # Generate a random hue between 0 and 360 degrees (unique color)
        hue = random.randint(0, 360)
        # Keep saturation and lightness high for brightness
        saturation = random.uniform(0.7, 1.0)  # 70% to 100%
        lightness = random.uniform(0.5, 0.7)  # 50% to 70%

        # Convert HSL to RGB using colorsys module
        import colorsys
        r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)

        # Convert RGB values to hex format for matplotlib
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(hex_color)

    return colors

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


def Fitness(cities, potential = 1500):

    """Calculates the total distances of a path between all the points in an ordered list of points.
    The last step goes from the last point in the list to the first. The potential barrier is added multiplying the distance
    whenever the path crosses the barrier value that is potential.

    Parameters:
            cities (list) : list of points needed to travel, in random order
        Returns:
            totDist: total distance between the cities in the given order"""

    # Initialize the distance in order to sum progressively all the distances
    TotDistance = 0

    # Iterate over the cities and calculate the total distance
    for i in range(len(cities)):
        startCity = cities[i]
        endCity = None

        if i + 1 < len(cities):
            endCity = cities[i + 1]
        else:
            endCity = cities[0]

        if (startCity[0] > potential and endCity[0] > potential) or (startCity[0] < potential and endCity[0] < potential):
            TotDistance += distance(startCity, endCity)
        else:
            TotDistance += distance(startCity, endCity)*1000


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


def plotta(cities, colore="y", title="Route", show=True):
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
        plt.plot(xvalues, yvalues, '--', color=colore)
        #plt.set_xlabel('X Coordinates')
        #ax.set_ylabel('Y Coordinates')
        plt.title(title)

        for city in cities:
            plt.scatter(city[0], city[1], color='red')
        plt.show()

    # Might need only the ordered values without the plot (Not used here)
    if not show:
        return xvalues, yvalues

def annealing(initial, initialTemp = 1, Nswap = 10, potential = 1500):
    """
    This function implements the simulated annealing algorithm. In this algorithm
    Parameters:
        initial (list) : input list of cities
        potential (float) : x position of the vertical line describing the potential barrier
    Returns:

         solution (list) : output list of cities after simulated annealing algorithm

         totalDistance (float) : total distance of the path between the cities in the order given by solution

    """
    T = initialTemp
    solution = initial
    sameDistance = 0
    sameSolution = 0

    for _ in range(Nswap): #Chooses a random neighbour
        number = randint(1,2)
        if number == 1:
            neighbour = or_opt(solution[:])
        elif number ==2:
            neighbour = inverse(solution[:])
        elif number == 3:
            neighbour = swap_routes(solution[:])
        elif number == 4:
            neighbour = two_opt(solution[:])


        distanceDiff = - Fitness(neighbour, potential) + Fitness(solution[:], potential) #Compares the total distances of the two solutions

        if distanceDiff > 0: #Accept if the neighbour has a better solution
            solution = neighbour
            sameDistance = 0
            sameSolution = 0

        elif distanceDiff == 0:
            sameDistance += 1
            sameSolution += 1
        # Now let's see if the solution is not better

        elif distanceDiff < 0:
            if random.gauss(0.5,0.005) <= np.exp((distanceDiff) / T): #Accept the change with a probability p
                solution = neighbour
                sameDistance = 0
                sameSolution = 0
            else:
                sameDistance += 1
                sameSolution += 1

    #print("The final result is:", Fitness(solution))

    return solution, Fitness(solution, potential), T


def PT(routes, temps, potential = 1500):
    """Implements the Parallel Tempering algorithm. Two systems that have temperatures close to each other are
    taken into account and a swap of temperatures between the copies is proposed with an acceptance ratio
    based on a Metropolis algorithm. This is done for all the systems with their neighbour system at an above temperature.
    Parameters:
          routes (list) : list of lists of the current routes in the copies of the system
          temps (list) : list of the temperatures associated with the different copies of the system.
          potential (float) : x value vertical line describing the potential barrier

    Returns:
          routes: the list of routes of the systems after the trial swaps
          temps: the list of temperatures associated with the routes after the swaps"""
    for i in range(len(temps)-1):
        DeltaT = 1/(temps[i+1]) - 1/temps[i]
        DeltaE =  Fitness(routes[i+1], potential) - Fitness(routes[i], potential)
        if uniform(0,1) < np.exp(DeltaT*DeltaE):
            temps[i], temps[i+1] = temps[i+1], temps[i]
    return routes, temps


def simulationFinal(Ntot, Nswap, Temps, paths, potential = 1500):
    """Performs the simulation of the systems. The idea is that it takes a number N of systems with different temperatures,
    then for each of them. The systems are evolved through a Metropolis algorithm for Nswaps times, and after that a swap is
    proposed with a Parallel Tempering algorithm (PT). this process is repeated for Ntot/Nswap times so that the
    total number of steps will be Ntot.
    Parameters:
          Ntot (int) : total number of simulation steps
          Nswap (int) : total number of Metropolis steps at each iteration
          Temps (list) : ordered list of temperatures of the systems
          paths (list) : list of the current paths between the cities in the copies of the simulation. They will all be the same at the beginning"""
    tempForPlot = []
    tempForPlot.append(Temps[:])

    start = time.time()

    for _ in range(int(Ntot / Nswap)):  # This will be the number of swap trials with PT

        routes = []

        # With annealing we let the system evolve
        for i in range(len(Temps)):
            bestRoute, bestRouteDistance, T = annealing(paths[i], initialTemp=Temps[i], Nswap=Nswap, potential = potential)
            routes.append(bestRoute[:])
            Temps[i] = T

        # PT will propose the temperature swaps
        paths, Temps = PT(routes, Temps, potential)

        print(_)  # Just prints the numbers to see how is it going

        tempForPlot.append(Temps[:])  # This will be used in the plots later on

    end = time.time()
    timeConvergence = end - start
    print("The necessary time is:", timeConvergence) #Let's check the time needed for the simulation


    return paths, Temps, tempForPlot





# Read the file with the city points

path = "lin105.tsp"
cities = read_tsp_coordinates(path)
scale = np.sqrt(5*len(cities))
distances = []


# Give the parameters for the simulations
potential = 1500
Temps = [0.0025, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
paths = [cities]*len(Temps)
initDistance = Fitness(cities, potential)
Ntot = 100000
Nswap = 100000/500


print("The initial distance is: ", initDistance) # We can see at which total distance do we start

# PERFORM THE SIMULATION
paths, Temps, tempForPlot = simulationFinal(int(Ntot), int(Nswap), Temps, paths, potential)




### ALL THAT COMES AFTER IS FOR PLOTTING AND SHOWING THE RESULTS ###



y_values_transposed = np.array(tempForPlot).T  # Needed for the systems swap plot

plt.figure(figsize=(12, 6))

colors = colors(len(tempForPlot[1])) # Get as many colors as the number of system copies (aka as many T we have)

# Now let's plot the swaps of temperatures between the systems
# Each line is a certain temperature while each color is a system. The colors are randomly generated by the function colors
xPlot = np.linspace(0, int(Ntot/Nswap), int(Ntot/Nswap))

for i in range(len(tempForPlot)-1):
    j = 0
    for element in tempForPlot[i]:
        plt.scatter(xPlot[i], element, color = colors[j],  marker='o', label=f'Line {i + 1}', linewidth=1)
        j += 1

# Add labels and title
plt.xlabel('Number of swap trials')
plt.ylabel('List of 9 temperatures')
plt.title('Temperature swaps')
plt.show()


# Find the best overall route and plot it at the end
distances = []
for element in paths:
    distances.append(Fitness(element[:]))
i = distances.index(min(distances)) # This finds the minimum distance simulated
j = 0
for element in paths:
    if j == i:
        bestRoute = element

    j+=1

distance = distances[i]

print("Best Distance is:", distance) # Shows the result of the simulation
plt.axvline(x=potential, color='red', linestyle='--', linewidth=2, label=f'x = {potential}')
plotta(bestRoute, colore = "green", title = "Best found route")# Plots the best route
