import tsplib95
import numpy as np
import random
import math 
import matplotlib.pyplot as plt
import time

# Read the file and return the problem with the nodes and weights using the tsplib95 library.
def read_file(file_name):
    return tsplib95.load(file_name)

# Create an initial population of candidate solutions.
def create_population(problem, pop_size):
    cities = np.array(list(problem.get_nodes()))                     
    return [np.random.permutation(cities) for _ in range(pop_size)]

# Return the summed distance between each candidate solution. 
def calculate_fitess(tour, problem):
    total_cost = 0
    num_cities = len(tour)
    for i in range(num_cities -1):
        total_cost += problem.get_weight(tour[i], tour[i+1])
    # Add the distance between the last and first city.
    total_cost += problem.get_weight(tour[0], tour[num_cities-1])
    return total_cost

# Selects the solution with the lowest distance cost from the tournament of randomly selected candidates. 
def tournament_selection(population, problem, tournament_size):
    # Randomly selects tours (candidate solutions) from the popluation to make up the tournament.
    tournament = random.sample(population, tournament_size)
    # Calculate the distance cost of each tour in the tournament.    
    tournament_costs = [calculate_fitess(tour, problem) for tour in tournament]
    # Return the tour with the lowest distance cost.
    return tournament[tournament_costs.index(min(tournament_costs))]

def OX1(parent1, parent2):
    size = len(parent1)
    child1, child2 = [None]*size, [None]*size
    # Randomly select the start and end point within the range of the parent size.
    start, end = sorted(random.sample(range(size), 2))
    # Copy the segments between the start and end index into the corresponding child for each parent
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1] 

    # Every city in parent 1 not already into child 2
    remaining_for_child2 = [city for city in parent1 if city not in child2[start:end+1]]
    # Every city in parent 2 not already in child 1
    remaining_for_child1 = [city for city in parent2 if city not in child1[start:end+1]]
    
    # the current index will be at the index just after the second 'cut' point
    # The % operator will wrap back around to the first index to avoid going out of bounds of the array.
    current_index = ((end+1) % size)

    # Now iterate through and add the remaining cities to the first child. 
    for city in remaining_for_child1:
        child1[current_index] = city
        current_index = ((current_index + 1) % size)

    # Do the same for the second child
    current_index = ((end+1) % size)
    
    for city in remaining_for_child2:
        child2[current_index] = city
        current_index = ((current_index + 1) % size)
    # Convert to np array or it gives back an array of numpy.int64(value)'s instead of regular int representation 
    # [numpy.int64(1),numpy.int64(2), ...] instead of [1,2, ...]
    child1 = np.array(child1)
    child2 = np.array(child2)
    return child1, child2

def OX2(parent1, parent2):
    size = len(parent1)
    # Create two children with the same number of genes (cities) as the parent. 
    child1, child2 = [None]*size, [None]*size
    # Find 1/4 of the tour length.
    k = size // 4 
    # Generate and sort the k random indexes 
    # note: They don't need to be sorted but I only noticed while the algorithm was already running and stopping now is bad (on generation 22 488 to go its already been > 3 hours).
    indexes = sorted(random.sample(range(size), k))

    # For each child copy the genes from one parent at the selected index (p2 -> c1 and p1 -> c2)
    for index in indexes:
        child1[index] = parent2[index]
        child2[index] = parent1[index]

     # Every city in parent1 not already in the first child.
    remaining_for_child1 = [city for city in parent1 if city not in child1]
    # Every city in parent2 not already in the second child.
    remaining_for_child2 = [city for city in parent2 if city not in child2]

    current_index_c1 = 0
    current_index_c2 = 0
    for i in range(size):
        if child1[i] is None:
            child1[i] = remaining_for_child1[current_index_c1]
            current_index_c1 += 1
        if child2[i] is None: 
            child2[i] = remaining_for_child2[current_index_c2]
            current_index_c2 += 1

    child1 = np.array(child1)
    child2 = np.array(child2)
    return child1, child2

def mutate_swap_rand(child_to_mutate):
    length = len(child_to_mutate)
    mutated_child = child_to_mutate[:] 
    
    # Generate a random integer between 0 and the tour length
    first_index, second_index = sorted(random.sample(range(length), 2))
    # Make a copy so we avoid modifying the original.       
    temp = mutated_child[first_index]
    # Swap the cities at the randomly generated indexes.   
    mutated_child[first_index] = mutated_child[second_index]
    mutated_child[second_index] = temp
    return mutated_child

# Introduces mutations by selecting a section of cities in a tour and shuffles them.
# If the tour is [1 2 3 4 5 6 7 8], [4 5 6 7] may be selected and shuffled:
# [4 5 6 7] becomes [4 7 5 6] and is reinserted into the tour:
# [1 2 3 7 5 4 6  8] 
def scramble_mutation(child_to_mutate):
    length = len(child_to_mutate)
    mutated_child = child_to_mutate[:]

    # Select two distinct indices and sort them.
    i, j = sorted(random.sample(range(length), 2))
    # Select the segment of the tour to shuffle.
    segment = mutated_child[i:j+1]
    # Shuffle the segment.
    random.shuffle(segment)
    # Reinsert the shuffled segment back into the tour.
    mutated_child[i:j+1] = segment
    # Return the mutated tour.
    return mutated_child

# Selects the mutation operator to use based on a random number between 0 and 1.
def select_mutation(child_to_mutate):
    if random.random() >= 0.5:
        return mutate_swap_rand(child_to_mutate)
    else:
        return scramble_mutation(child_to_mutate)

# Selects the crossover operator to use based on a random number between 0 and 1.
def select_crossover(p1, p2):
    if random.random() >= 0.5:
        return OX1(p1,p2)
    else:
        return OX2(p1,p2)
    

def genetic_algorithm(problem, population_size, crossover_rate, mutation_rate, tournament_size, log):
    # A get the log.text file so we can log the results of the algorithm.
    print(f"Running Genetic Algorithm with {population_size} population size, {crossover_rate} crossover rate, {mutation_rate} mutation rate, and {tournament_size} tournament size\n")
    population = create_population(problem, population_size)
    all_generations = []

    no_evolution_counter = 0
    # Initialise as none, later replace with the best fit value for the best solution of this generation.
    best_fitness_overall = None
    
    start_time = time.time()
    # Run the algorithm for x generations
    for generation in range(500):
        if no_evolution_counter >= 150:
            log.write("Stopping early, no improvement in 150 consecutive generations")
            break
        print(generation)    
        # Implement Selection and Crossover 
        new_generation = []
        while len(new_generation) < population_size:
            # Select parents from the population using tournament selection.
            parent1 = tournament_selection(population, problem,tournament_size)
            parent2 = tournament_selection(population, problem, tournament_size)

            # Generate number between 0 and 1, and perform crossover if it is less than the crossover_rate. 
            if random.random() < crossover_rate:
                child1, child2 = select_crossover(parent1, parent2)
                # If crossover is performed, then give each offspring a chance to mutate based on the mutation rate. 
                if random.random() < mutation_rate:
                    child1 = select_mutation(child1)
                if random.random() < mutation_rate:
                    child2 = select_mutation(child2)
            else:
                # Otherwise copy the parents as the next generation.
                child1, child2 = parent1[:], parent2[:]

            
            # Add the children to the new population which will be the next generation (generational replacement).
            new_generation.extend([child1, child2])
        # Add this generation to all_generations to create graphs later.
        all_generations.extend([population])
        # Set the population to be the new generation
        population = new_generation

        # Calculate the best fitness of this generation, if it is not better than the previous generations best, 
        # Then increment the no_evolution_counter which is used as a stopping criterion for the sake of efficiency. 
        fitness_scores = [calculate_fitess(tour, problem) for tour in population]
        lowest_cost = min(fitness_scores)
        if best_fitness_overall is None or lowest_cost < best_fitness_overall:
            best_fitness_overall = lowest_cost
            no_evolution_counter = 0
        else:
            no_evolution_counter += 1
    # Calculate fitness for each member of the current population
    final_fitness_scores = [calculate_fitess(tour, problem) for tour in population]
    # Find the best candidate (with the lowest cost/fitness) and return the index with np.argmin.
    final_best_candidate = population[np.argmin(final_fitness_scores)]
    # Save the fitness for the best candidate with min()
    best_final_cost = min(final_fitness_scores)
    # Used to calculate the time taken to run the algorithm
    fin_time = time.time()
    # Write details to the log file.
    log.write(f"Final Generation Best Candidate: {final_best_candidate}\n"+
              f"Final Generation Best Fitness: {best_final_cost}\n"+
              f"Number of Generations: {len(all_generations)}\n"+
              f"Took {fin_time - start_time} Seconds to complete\n\n")
    return all_generations

def fitness_over_generations(all_generations, problem, num, path):
    avg_costs = []

    # For each generation, calculate the fitness of each tour, and find the 
    for generation in all_generations:
        fitness_scores = [calculate_fitess(tour, problem) for tour in generation]
        avg = sum(fitness_scores)/len(fitness_scores)
        avg_costs.append(avg)

    plt.figure(figsize=(10,6))
    plt.plot(avg_costs, marker='o', linestyle='-')
    plt.title("Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.grid(True)
    plt.savefig(path+str(num)+'.png')
    
############################################################################################################################################################################
# Run the algorithm for the berlin52.tsp file
berlin52 = "./data/berlin52.tsp"
berlin52_figure_path = './plots/berlin95_plots/'
problem = read_file(berlin52)
population_sizes = [100,500, 1000]
crossover_rates = [0.8,0.9,1.0]
mutation_rates = [0.01, 0.1, 0.2]
tournament_size = 5
log = open("log.txt", "a")

plot_num = 0
for pop_size in population_sizes:
    for cross_rate in crossover_rates:
        for mut_rate in mutation_rates:
           log.write(f"Running for Population Size: {pop_size}, Crossover Rate: {cross_rate}, Mutation Rate: {mut_rate} Tournament Size: {pop_size//20}\n")
           print(f"Running for Population Size: {pop_size}, Crossover Rate: {cross_rate}, Mutation Rate: {mut_rate} Tournament Size: {pop_size//20}\n")
           fitness_over_generations(genetic_algorithm(problem, pop_size, cross_rate, mut_rate, pop_size//20,log), problem, plot_num, berlin52_figure_path)
           plot_num += 1

log.write("Berlin Complete\n")

#############################################################################################################################################################################
# Run the algorithm for the kroA100.tsp file
kroA100 = "./data/kroA100.tsp"
kroA100_figure_path = './plots/kroA100_plots/'
problem = read_file("./data/kroA100.tsp")
population_sizes = [100, 500, 1000]
crossover_rates = [0.8,0.9, 1.0]
mutation_rates = [0.01, 0.1, 0.2]
log = open("log.txt", "a")

log.write('KroA100 Log\n')

plot_num = 0
for pop_size in population_sizes:
    for cross_rate in crossover_rates:
        for mut_rate in mutation_rates:
           log.write(f"Running for Population Size: {pop_size}, Crossover Rate: {cross_rate}, Mutation Rate: {mut_rate} Tournament Size: {pop_size//20}\n")
           print(f"Running for Population Size: {pop_size}, Crossover Rate: {cross_rate}, Mutation Rate: {mut_rate} Tournament Size: {pop_size//20}\n")
           fitness_over_generations(genetic_algorithm(problem, pop_size, cross_rate, mut_rate, pop_size//20,log), problem, plot_num, kroA100_figure_path)
           plot_num += 1

log.write("KroA100 Complete\n")
log.close()

#############################################################################################################################################################################
# Run the algorithm for the pr1002.tsp file
pr1002 = "./data/pr1002.tsp"
pr1002_figure_path = './plots/pr1002_plots/'
problem = read_file(pr1002)
population_sizes = [100, 500, 1000]
crossover_rates = [0.8,0.9, 1.0]
mutation_rates = [0.01, 0.1, 0.2]
log = open("log.txt", "a")
log.write('Pr1002 Log\n')

plot_num = 0
for pop_size in population_sizes:
    for cross_rate in crossover_rates:
        for mut_rate in mutation_rates:
           log.write(f"Running for Population Size: {pop_size}, Crossover Rate: {cross_rate}, Mutation Rate: {mut_rate} Tournament Size: {pop_size//20}\n")
           print(f"Running for Population Size: {pop_size}, Crossover Rate: {cross_rate}, Mutation Rate: {mut_rate} Tournament Size: {pop_size//20}\n")
           fitness_over_generations(genetic_algorithm(problem, pop_size, cross_rate, mut_rate, pop_size//20,log), problem, plot_num, pr1002_figure_path)
           plot_num += 1

log.write("Pr1002 Complete\n")
log.close()