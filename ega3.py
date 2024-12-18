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
                penalty += 2500  # Kara za przecięcie przeszkody
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


def pmx_crossover(parent1, parent2):
    """Krzyżowanie PMX - wymiana fragmentów tras między dwoma rodzicami."""
    # Wybieramy dwa losowe punkty (indeksy)
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1, len(parent2) - 1)

    # Tworzymy potomków jako kopie rodziców
    child1 = parent1[:]
    child2 = parent2[:]

    # Zamieniamy fragmenty tras między rodzicami
    for i in range(point1, point2 + 1):
        child1[i], child2[i] = child2[i], child1[i]

    # Naprawiamy duplikaty w dzieciach
    child1 = repair(child1, parent1, parent2, point1, point2)
    child2 = repair(child2, parent1, parent2, point1, point2)

    return child1, child2


def repair(child, parent1, parent2, point1, point2):
    """Naprawia duplikaty w dziecku po wymianie fragmentów."""
    for i in range(len(child)):
        if i < point1 or i > point2:
            # Jeśli element z dziecka jest duplikatem, naprawiamy go
            if child[i] in parent1[point1:point2 + 1]:
                child[i] = find_missing_gene(child[i], parent1, parent2, point1, point2)
            elif child[i] in parent2[point1:point2 + 1]:
                child[i] = find_missing_gene(child[i], parent1, parent2, point1, point2)
    return child


def find_missing_gene(gene, parent1, parent2, point1, point2):
    """Znajduje brakujący gen w dziecku, który nie występuje w wybranym fragmencie."""
    for i in range(len(parent1)):
        if parent1[i] not in parent2[point1:point2 + 1] and parent2[i] not in parent1[point1:point2 + 1]:
            return parent1[i] if parent1[i] != gene else parent2[i]
    return gene


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


def enhanced_genetic_algorithm(start, end, obstacles, num_generations=100, population_size=50, num_points=5,
                               mutation_rate=0.1):
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
            child1, child2 = pmx_crossover(parent1, parent2)  # Zastosowanie PMX
            new_population.append(mutate(child1, mutation_rate, bounds))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate, bounds))

        population = new_population

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
    plt.title(f'Pokolenie {population_size}')
    plt.show()

    return best_path, best_fitness_history, best_population_history


def plot_path(path, obstacles, start, end, population_size):
    """Wizualizacja trasy."""
    plt.figure()
    for (x1, y1, x2, y2) in obstacles:
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray'))
    x_vals, y_vals = zip(*path)
    plt.plot(x_vals, y_vals, marker='o', color='blue', label='Path')
    plt.plot(*start, marker='o', color='green', label='Start')
    plt.plot(*end, marker='o', color='red', label='End')
    plt.title(f'Pokolenie {population_size}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Przykład użycia:
    start = (1, 1)
    end = (99, 99)
    obstacles1 = [(30, 30, 50, 50), (60, 10, 70, 40), (20, 60, 40, 80)]
    generation_number = 1000
    population_size1 = 25
    population_size2 = 50

    for population_size in [population_size1, population_size2]:
        best_path, best_fitness_history, best_population_history = enhanced_genetic_algorithm(start=start, end=end,
                                                                                              obstacles=obstacles1,
                                                                                              num_generations=generation_number,
                                                                                              population_size=population_size)
        plot_path(best_path, obstacles1, start, end, population_size)
