import numpy as np
import random

import matplotlib.pyplot as plt

# Parametry
population_size = 100      # Liczba osobników w populacji
mutation_rate = 0.2        # Prawdopodobieństwo mutacji
num_generations = 100     # Liczba generacji
elite_size = 2             # Liczba elitarnych osobników, które przechodzą do kolejnej generacji
grid_size = (20, 20)       # Rozmiar mapy (np. 20x20)
start_point = (0, 0)       # Punkt startowy
end_point = (7, 10)       # Punkt końcowy

# Przykładowe przeszkody
#obstacles = {(5, 5), (7, 6),  (6, 4), (10, 10), (17, 18)}
obstacles = {
                (5,6),(5,7),(5,8),(5,9),(5,10)
                ,(6,6),(6,7),(6,8),(6,9),(6,10)


            }

# Funkcja tworzenia początkowej populacji
def create_population():
    population = []
    for _ in range(population_size):
        path = [start_point]
        while path[-1] != end_point:
            x, y = path[-1]
            
            # Celowanie w kierunku punktu końcowego
            dx = np.sign(end_point[0] - x)
            dy = np.sign(end_point[1] - y)
            
            # Wybieranie następnego kroku w kierunku celu lub losowego
            next_step = (x + dx * random.choice([0, 1]), y + dy * random.choice([0, 1]))
            next_step = (
                max(0, min(next_step[0], grid_size[0] - 1)),
                max(0, min(next_step[1], grid_size[1] - 1))
            )
            
            # Sprawdzenie kolizji z przeszkodami i duplikacji punktu
            if next_step not in path and next_step not in obstacles:
                path.append(next_step)
            
            # Bezpieczeństwo: ograniczenie liczby kroków
            if len(path) > grid_size[0] * grid_size[1]:  # Max liczba kroków
                break
        
        # Dodaj ścieżkę tylko, jeśli osiągnęła punkt końcowy
        if path[-1] == end_point:
            population.append(path)
    return population


# Funkcja oceny ścieżki
def fitness(path):
    distance = len(path)
    collisions = sum(1 for p in path if p in obstacles)
    return distance + collisions * 100  # Kara za kolizje

# Selekcja ruletkowa z dodatkowym zabezpieczeniem
def selection(population):
    if len(population) <= elite_size:
        return population  # Zwracamy całą populację, jeśli jest za mała na selekcję
    
    scores = [1 / (1 + fitness(individual)) for individual in population]
    total = sum(scores)
    probs = [score / total for score in scores]
    
    # Bezpieczne losowanie osobników z prawdopodobieństwem
    try:
        selected = np.random.choice(population, size=population_size - elite_size, p=probs).tolist()
    except ValueError:
        selected = random.sample(population, population_size - elite_size)  # Wybierz losowo, jeśli błąd
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

# Wizualizacja ścieżki
plt.figure(figsize=(8, 8))
plt.scatter(*zip(*obstacles), color='red', s=100)
plt.plot(*zip(*best_path), color='blue', marker='o')
plt.scatter(*zip(*best_path), color='blue', s=100)
plt.scatter(*start_point, color='green', s=100)
plt.scatter(*end_point, color='green', s=100)
plt.xlim(-1, grid_size[0])
plt.ylim(-1, grid_size[1])
plt.gca().invert_yaxis()
plt.show()

