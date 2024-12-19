import numpy as np
import random
import matplotlib.pyplot as plt
from numba import jit, prange

# Definicja parametrów algorytmu
POPULATION_SIZE = 100
CHROMOSOME_LENGTH = 20
CROSSOVER_RATE = 1
MUTATION_RATE = 0.8
MAX_GENERATIONS = 10000

# Funkcja sprawdzająca kolizję z przeszkodami
@jit(nopython=True)
def is_collision(point, obstacles):
    x, y = point
    for obs in obstacles:
        if obs[0] <= x <= obs[2] and obs[1] <= y <= obs[3]:
            return True
    return False

# Funkcja obliczająca przystosowanie
@jit(nopython=True)
def fitness_function(chromosome, start, end, obstacles):
    penalty = 0
    total_distance = 0
    smoothness_penalty = 0
    path = np.zeros((len(chromosome) + 2, 2), dtype=np.float64)
    path[0] = np.array(start, dtype=np.float64)
    path[-1] = np.array(end, dtype=np.float64)
    for i in range(len(chromosome)):
        path[i + 1] = chromosome[i]

    for i in range(len(path) - 1):
        if is_collision(path[i], obstacles) or is_collision(path[i + 1], obstacles):
            penalty += 1e8

        total_distance += np.linalg.norm(path[i] - path[i + 1])

        if i > 0:
            v1 = path[i] - path[i - 1]
            v2 = path[i + 1] - path[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            smoothness_penalty += (1 - cos_angle)

    distance_to_goal = np.linalg.norm(path[-1] - np.array(end, dtype=np.float64))
    fitness = 1 / (total_distance + penalty + smoothness_penalty + distance_to_goal + 1e-6)
    return fitness

# Selekcja turniejowa
@jit(nopython=True)
def tournament_selection(population, fitnesses):
    selected = np.empty(population.shape, dtype=population.dtype)
    for idx in range(len(population)):
        i, j = np.random.randint(0, len(population), 2)
        if fitnesses[i] > fitnesses[j]:
            selected[idx] = population[i]
        else:
            selected[idx] = population[j]
    return selected

# Krzyżowanie
@jit(nopython=True)
def crossover(parent1, parent2):
    if np.random.random() < CROSSOVER_RATE:
        points = np.random.choice(CHROMOSOME_LENGTH, 2, replace=False)
        point1, point2 = min(points), max(points)
        child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        return child1, child2
    return parent1, parent2

# Mutacja
@jit(nopython=True)
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        point = random.randint(0, CHROMOSOME_LENGTH - 1)
        chromosome[point] = (random.randint(0, 100), random.randint(0, 100))
    return chromosome

# Funkcja inicjalizująca populację
def initialize_population():
    population = np.empty((POPULATION_SIZE, CHROMOSOME_LENGTH, 2), dtype=np.int64)
    for i in range(POPULATION_SIZE):
        for j in range(CHROMOSOME_LENGTH):
            population[i, j] = (random.randint(0, 100), random.randint(0, 100))
    return population

# Algorytm główny
def enhanced_genetic_algorithm(start, end, obstacles):
    population = initialize_population()
    best_solution = None
    best_fitness = -np.inf
    best_fitnesses = []
    average_fitnesses = []

    for generation in range(MAX_GENERATIONS):
        fitnesses = np.array([fitness_function(chrom, start, end, obstacles) for chrom in population])

        for i, fit in enumerate(fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_solution = population[i]

        best_fitnesses.append(best_fitness)
        average_fitnesses.append(np.mean(fitnesses))

        selected = tournament_selection(population, fitnesses)

        next_generation = np.empty_like(population)
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[min(i+1, len(selected)-1)]
            child1, child2 = crossover(parent1, parent2)
            next_generation[i] = mutate(child1)
            next_generation[i+1] = mutate(child2)

        population = next_generation
        print(f"Pokolenie {generation}: Najlepsze fitness = {best_fitness}")

    plt.figure()
    plt.plot(best_fitnesses, label='Best Fitness')
    plt.plot(average_fitnesses, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

    return best_solution

# Funkcja rysująca trasę robota
def plot_path(start, end, solution, obstacles):
    plt.figure(figsize=(8, 8))
    plt.scatter(*start, color='green', label='Start')
    plt.scatter(*end, color='red', label='End')
    for obs in obstacles:
        plt.gca().add_patch(plt.Rectangle((obs[0], obs[1]), obs[2]-obs[0], obs[3]-obs[1], color='gray', alpha=0.5))
    path = [start] + solution + [end]
    plt.plot(*zip(*path), marker='o', color='blue', label='Path')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid()
    plt.show()

# Test programu
if __name__ == "__main__":
    start = (10, 10)
    end = (90, 90)
    obstacles = [
        (30, 30, 50, 50),
        (60, 10, 70, 40),
        (20, 60, 40, 80)
    ]

    solution = enhanced_genetic_algorithm(start, end, obstacles)
    plot_path(start, end, solution, obstacles)
