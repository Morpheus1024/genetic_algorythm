import numpy as np
import random

# Parametry
population_size = 100      # Liczba osobników w populacji
mutation_rate = 0.1        # Prawdopodobieństwo mutacji
num_generations = 1000     # Liczba generacji
elite_size = 5             # Liczba elitarnych osobników, które przechodzą do kolejnej generacji
grid_size = (20, 20)       # Rozmiar mapy (np. 20x20)
start_point = (0, 0)       # Punkt startowy
end_point = (19, 19)       # Punkt końcowy

# Przykładowe przeszkody
obstacles = {(5, 5), (5, 6), (6, 5), (10, 10), (15, 15)}

# Funkcja tworzenia początkowej populacji
def create_population():
    population = []
    for _ in range(population_size):
        path = [start_point]
        while path[-1] != end_point:
            x, y = path[-1]
            next_step = (x + random.choice([-1, 0, 1]), y + random.choice([-1, 0, 1]))
            next_step = (max(0, min(next_step[0], grid_size[0] - 1)),
                         max(0, min(next_step[1], grid_size[1] - 1)))
            if next_step not in path and next_step not in obstacles:
                path.append(next_step)
        population.append(path)
    return population

# Funkcja oceny ścieżki
def fitness(path):
    distance = len(path)
    collisions = sum(1 for p in path if p in obstacles)
    return distance + collisions * 100  # Kara za kolizje

# Selekcja ruletkowa
def selection(population):
    scores = [1 / (1 + fitness(individual)) for individual in population]
    total = sum(scores)
    probs = [score / total for score in scores]
    selected = np.random.choice(population, size=population_size - elite_size, p=probs).tolist()
    return selected

# Operator krzyżowania
def crossover(parent1, parent2):
    split = random.randint(1, min(len(parent1), len(parent2)) - 2)
    child = parent1[:split] + [p for p in parent2 if p not in parent1[:split]]
    return child

# Operator mutacji
def mutate(path):
    if random.random() < mutation_rate:
        idx = random.randint(1, len(path) - 2)
        x, y = path[idx]
        new_step = (x + random.choice([-1, 0, 1]), y + random.choice([-1, 0, 1]))
        new_step = (max(0, min(new_step[0], grid_size[0] - 1)),
                    max(0, min(new_step[1], grid_size[1] - 1)))
        if new_step not in obstacles and new_step not in path:
            path[idx] = new_step
    return path

# Główna pętla algorytmu EGA
population = create_population()
for generation in range(num_generations):
    # Sortowanie populacji według funkcji oceny
    population = sorted(population, key=fitness)
    
    # Elitaryzm: zachowanie najlepszych osobników
    new_population = population[:elite_size]
    
    # Tworzenie nowej generacji poprzez selekcję, krzyżowanie i mutację
    selected = selection(population)
    for i in range(0, len(selected), 2):
        if i + 1 < len(selected):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))
    
    # Uzupełnianie populacji do żądanej wielkości
    population = new_population[:population_size]

    # Wyświetlanie informacji o najlepszym osobniku w każdej generacji
    best_path = population[0]
    best_fitness = fitness(best_path)
    print(f"Generation {generation}: Best Fitness = {best_fitness}, Path Length = {len(best_path)}")

# Wyświetlenie najlepszej znalezionej ścieżki
print("Best Path Found:", best_path)