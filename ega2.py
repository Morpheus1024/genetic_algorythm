import random
import matplotlib.pyplot as plt
import numpy as np

def initialize_population(start, end, num_individuals, num_points):
    """Generuje początkową populację tras."""
    population = []
    for _ in range(num_individuals):
        path = [start] + [
            (random.uniform(min(start[0], end[0]), max(start[0], end[0])),
             random.uniform(min(start[1], end[1]), max(start[1], end[1])))
            for _ in range(num_points)
        ] + [end]
        population.append(path)
    return population

def fitness_function(path, obstacles):
    """Ocena trasy: długość + kara za przecięcia przeszkód."""
    total_distance = 0
    penalty = 0
    for i in range(len(path) - 1):
        total_distance += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
        for obs in obstacles:
            if line_intersects_obstacle(path[i], path[i + 1], obs):
                penalty += 1000  # Kara za przecięcie przeszkody
    return total_distance + penalty

def line_intersects_obstacle(p1, p2, obstacle):
    """Sprawdza, czy linia między p1 a p2 przecina przeszkodę."""
    x1, y1, x2, y2 = obstacle
    px1, py1 = p1
    px2, py2 = p2
    rectangle_lines = [
        ((x1, y1), (x2, y1)),  # Dolna krawędź
        ((x2, y1), (x2, y2)),  # Prawa krawędź
        ((x2, y2), (x1, y2)),  # Górna krawędź
        ((x1, y2), (x1, y1)),  # Lewa krawędź
    ]
    for rect_line in rectangle_lines:
        if check_line_intersection(p1, p2, rect_line[0], rect_line[1]):
            return True
    return False

def check_line_intersection(p1, p2, q1, q2):
    """Sprawdza, czy dwie linie się przecinają."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def crossover(parent1, parent2):
    """Krzyżowanie dwojga rodziców - wielopunktowe."""
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1, len(parent2) - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

def mutate(individual, mutation_rate, bounds):
    """Mutacja osobnika - z dynamicznym mutowaniem."""
    for i in range(1, len(individual) - 1):
        if random.random() < mutation_rate:
            individual[i] = (
                random.uniform(bounds[0][0], bounds[1][0]),
                random.uniform(bounds[0][1], bounds[1][1]),
            )
    return individual

def select_parents(population, fitnesses, tournament_size=3):
    """Selekcja turniejowa z adaptacyjnym rozmiarem."""
    parents = []
    for _ in range(2):  # Selekcja dwóch rodziców
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)
        parents.append(tournament[0][0])
    return parents

def elitism(population, fitnesses, elitism_size=2):
    """Zachowanie najlepszych osobników (elitarność)."""
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]
    return sorted_population[:elitism_size]

def enhanced_genetic_algorithm(start, end, obstacles, num_generations=100, population_size=50, num_points=5, mutation_rate=0.1):
    """Algorytm genetyczny EGA do planowania drogi z elitarnością, adaptacyjnymi parametrami i metodami selekcji."""
    population = initialize_population(start, end, population_size, num_points)
    bounds = [(min(start[0], end[0]), min(start[1], end[1])),
              (max(start[0], end[0]), max(start[1], end[1]))]

    best_fitness_history = []  # Historia najlepszych fitnessów
    best_population_history = []  # Historia najlepszych populacji

    for generation in range(num_generations):
        # Dynamiczna zmiana parametrów
        mutation_rate = max(0.05, mutation_rate * 0.99)  # Spadek prawdopodobieństwa mutacji
        tournament_size = max(3, int(population_size / 10))  # Adaptacja rozmiaru turnieju

        # Oblicz funkcję fitness
        fitnesses = [1 / (fitness_function(path, obstacles) + 1e-6) for path in population]

        # Elitarność - zachowanie najlepszych osobników
        elites = elitism(population, fitnesses)

        # Nowa populacja
        new_population = elites
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitnesses, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate, bounds))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate, bounds))

        population = new_population

        # Opcjonalnie, hybrydyzacja z lokalnym przeszukiwaniem
        if generation % 10 == 0:
            best_path = min(zip(population, fitnesses), key=lambda x: x[1])[0]
            # Tu można dodać lokalne ulepszanie najlepszej trasy (np. algorytm najbliższego sąsiada)

        # Śledzenie najlepszej trasy i najlepszego fitnessu
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Dodajemy do historii najlepszą populację
        best_population_index = fitnesses.index(best_fitness)
        best_population_history.append(population[best_population_index])

    # Wybór najlepszego rozwiązania
    best_path = min(zip(population, fitnesses), key=lambda x: x[1])[0]

    # Wizualizacja zmian najlepszych fitnessów
    plt.plot(best_fitness_history)
    plt.xlabel('Generacja')
    plt.ylabel('Najlepszy fitness')
    plt.title('Ewolucja najlepszego fitnessu')
    plt.show()

    return best_path, best_fitness_history, best_population_history

def plot_path(path, obstacles, start, end):
    """Wizualizacja trasy."""
    plt.figure()
    for (x1, y1, x2, y2) in obstacles:
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray'))
    x_vals, y_vals = zip(*path)
    plt.plot(x_vals, y_vals, marker='o', color='blue')
    plt.scatter(*start, color='green', label='Start')
    plt.scatter(*end, color='red', label='End')
    plt.legend()
    plt.show()

# Przykład użycia:
start = (10, 10)
end = (90, 90)
obstacles = [(30, 30, 50, 50), (60, 10, 70, 40), (20, 60, 40, 80)]
best_path, best_fitness_history, best_population_history = enhanced_genetic_algorithm(start, end, obstacles)

# Wizualizacja trasy
plot_path(best_path, obstacles, start, end)
