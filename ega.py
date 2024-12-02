import numpy as np
import random
import matplotlib.pyplot as plt

# Definicja parametrów algorytmu
POPULATION_SIZE = 100  # Wielkość populacji (dostosowanie do różnych problemów)
CHROMOSOME_LENGTH = 20  # Maksymalna liczba punktów na trasie
CROSSOVER_RATE = 0.8  # Prawdopodobieństwo krzyżowania (wysoka eksploracja)
MUTATION_RATE = 0.2  # Prawdopodobieństwo mutacji (zachowanie różnorodności)
MAX_GENERATIONS = 10000  # Maksymalna liczba pokoleń

# Funkcja sprawdzająca kolizję z przeszkodami
def is_collision(point, obstacles):
    x, y = point
    for obs in obstacles:
        # Sprawdzanie, czy punkt (x, y) znajduje się w granicach przeszkody
        if obs[0] <= x <= obs[2] and obs[1] <= y <= obs[3]:
            return True
    return False

# Funkcja obliczająca przystosowanie (rozbudowana analiza z tekstu)
def fitness_function(chromosome, start, end, obstacles):
    """
    Funkcja fitness ocenia jakość trasy na podstawie:
    - Całkowitej długości trasy.
    - Kolizji z przeszkodami (wysokie kary za wejście w przeszkodę).
    - Gładkości trasy (minimalizacja kąta zmiany kierunku).
    - Bliskości końcowego punktu do celu.
    """
    penalty = 0  # Kara za kolizje
    total_distance = 0  # Całkowita odległość
    smoothness_penalty = 0  # Kara za brak gładkości
    path = [start] + chromosome + [end]

    for i in range(len(path) - 1):
        # Sprawdzenie kolizji dla każdego punktu
        if is_collision(path[i], obstacles) or is_collision(path[i+1], obstacles):
            penalty += 1e6  # Wysoka kara za kolizję

        # Długość odcinka między dwoma punktami
        total_distance += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))

        # Gładkość: kara za ostre kąty między trzema kolejnymi punktami
        if i > 0:
            v1 = np.array(path[i]) - np.array(path[i - 1])  # Wektor 1
            v2 = np.array(path[i + 1]) - np.array(path[i])  # Wektor 2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)  # Kosinus kąta
            smoothness_penalty += (1 - cos_angle)  # Kara rośnie dla ostrych kątów

    # Kara za odległość końcowego punktu od celu (w przypadku niedokładnego trafienia)
    distance_to_goal = np.linalg.norm(np.array(path[-1]) - np.array(end))

    # Funkcja fitness: Wyższa wartość oznacza lepszą trasę
    fitness = 1 / (total_distance + penalty + smoothness_penalty + distance_to_goal + 1e-6)
    return fitness

# Selekcja turniejowa (zaawansowana metoda selekcji)
def tournament_selection(population, fitnesses):
    """
    Turniej wybiera dwóch losowych osobników i zwraca lepszego z nich.
    """
    selected = []
    for _ in range(len(population)):
        i, j = random.sample(range(len(population)), 2)
        # Wybór lepszego osobnika
        selected.append(population[i] if fitnesses[i] > fitnesses[j] else population[j])
    return selected

# Krzyżowanie dwupunktowe (rozbudowany operator krzyżowania)
def crossover(parent1, parent2):
    """
    Dwupunktowe krzyżowanie:
    - Dwa losowe punkty dzielące chromosomy rodziców.
    - Powstają nowe potomki z wymianą fragmentów.
    """
    if random.random() < CROSSOVER_RATE:
        point1, point2 = sorted(random.sample(range(CHROMOSOME_LENGTH), 2))
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2
    return parent1, parent2

# Mutacja punktowa (zachowanie różnorodności genetycznej)
def mutate(chromosome):
    """
    Mutacja losowo zmienia jeden punkt w chromosomie:
    - Pomaga unikać zbieżności do suboptymalnych rozwiązań.
    """
    if random.random() < MUTATION_RATE:
        point = random.randint(0, CHROMOSOME_LENGTH - 1)
        chromosome[point] = (random.randint(0, 100), random.randint(0, 100))
    return chromosome

# Algorytm główny (koordynacja EGA)

def initialize_population():
    """
    Inicjalizuje populację jako listę chromosomów.
    """
    population = []
    for _ in range(POPULATION_SIZE):
        # Generowanie losowych punktów (x, y) w przestrzeni
        chromosome = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(CHROMOSOME_LENGTH)]
        population.append(chromosome)
    return population


def enhanced_genetic_algorithm(start, end, obstacles):
    """
    - Inicjalizacja populacji.
    - Iteracyjny proces selekcji, krzyżowania, mutacji.
    - Elityzm: zachowanie najlepszego rozwiązania.
    """
    population = initialize_population()
    best_solution = None
    best_fitness = -np.inf

    for generation in range(MAX_GENERATIONS):
        # Obliczanie przystosowania dla całej populacji
        fitnesses = [fitness_function(chrom, start, end, obstacles) for chrom in population]

        # Aktualizacja najlepszego rozwiązania (elityzm)
        for i, fit in enumerate(fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_solution = population[i]

        # Selekcja lepszych osobników
        selected = tournament_selection(population, fitnesses)

        # Tworzenie nowego pokolenia
        next_generation = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[min(i+1, len(selected)-1)]
            # Krzyżowanie
            child1, child2 = crossover(parent1, parent2)
            # Mutacja
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))

        population = next_generation

        # Wyświetlanie postępu
        print(f"Pokolenie {generation}: Najlepsze fitness = {best_fitness}")

    return best_solution

# Funkcja rysująca trasę robota
def plot_path(start, end, solution, obstacles):
    """
    Wizualizacja środowiska:
    - Start, meta, przeszkody i trasa robota.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(*start, color='green', label='Start')
    plt.scatter(*end, color='red', label='End')
    for obs in obstacles:
        # Rysowanie przeszkód jako prostokątów
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
    # Punkt startowy i końcowy
    start = (10, 10)
    end = (90, 90)

    # Przeszkody (prostokąty definiowane jako [x1, y1, x2, y2])
    obstacles = [
        (30, 30, 50, 50),
        (60, 10, 70, 40),
        (20, 60, 40, 80)
    ]

    # Uruchomienie algorytmu
    solution = enhanced_genetic_algorithm(start, end, obstacles)

    # Wizualizacja trasy
    plot_path(start, end, solution, obstacles)
