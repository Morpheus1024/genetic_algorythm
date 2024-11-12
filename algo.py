import lib
import random
import numpy as np
import matplotlib.pyplot as plt

# Parametry
population_size = 100      # Liczba osobników w populacji
mutation_rate = 0.5        # Prawdopodobieństwo mutacji
num_generations = 100     # Liczba generacji
elite_size = 2             # Liczba elitarnych osobników, które przechodzą do kolejnej generacji
grid_size = (20, 20)       # Rozmiar mapy (np. 20x20)
start_point = (0, 0)       # Punkt startowy
end_point = (17, 16)       # Punkt końcowy

# Przykładowe przeszkody
#obstacles = {(5, 5), (7, 6),  (6, 4), (10, 10), (17, 18)}
#obstacles = {(5,6),(5,7),(5,8),(5,9),(5,10)}

#DONE: Dodać zaciąganie przeszkód z pliku
#WIP: Przekleić funkcje do lib.py
#TODO: Sprawdzenie, czy endpoint nie jest otoczony i czy nie jestet przeszkodą
#TODO: Dodać badanie wyboru trasy dla zmieniających się parametrów np. mutation rate += 0.1 i wyświetlić w celu porównaniu
#TODO: W README dodać opis EGA i opis programu

obstacles = {
    (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),  # pionowy blok
    (6, 1), (6, 2), (6, 3), (6, 4),         # krótki poziomy blok
    (8, 8), (8, 9), (8, 10), (8, 11),       # kolejny poziomy blok
    (10, 3), (11, 3), (12, 3), (13, 3),     # pionowy blok
    (15, 15), (15, 16), (15, 17),           # krótki poziomy blok
    (18, 5), (17, 5), (16, 5), (15, 5),     # kolejny poziomy blok
    (10, 10), (11, 10), (12, 10),           # centralny poziomy blok
    (5, 13), (5, 14), (5, 15), (5, 16),     # kolejny pionowy blok
    (7, 17), (8, 17), (9, 17),              # poziomy blok
    (18, 18), (17, 17), (16, 16), (15, 15), # ukośny blok
    (13, 13), (12, 12), (11, 11), (10, 10), # ukośny blok
    (4, 8), (5, 8), (6, 8),                 # krótki poziomy blok
    # (14, 14), (14, 13), (14, 12),           # krótki pionowy blok
    #(9, 2), (10, 2), (11, 2),               # blok na dole
    #(2, 17), (3, 17), (4, 17), (5, 17)      # blok na górze
}

obstacles = lib.read_obstacles_from_file('obstacles.txt')


# Funkcja oceny ścieżki
def fitness(path):
    distance = len(path)
    collisions = sum(1 for p in path if p in obstacles)
    return distance + collisions * 100  # Kara za kolizje

# Selekcja ruletkowa z dodatkowym zabezpieczeniem
def selection(population):
    if len(population) <= elite_size:
        return population  # Zwracamy całą populację, jeśli jest za mała na selekcję
    
    scores = [1 / (1 + lib.fitness(individual)) for individual in population]
    total = sum(scores)
    probs = [score / total for score in scores]
    
    # Bezpieczne losowanie osobników z prawdopodobieństwem
    try:
        selected = np.random.choice(population, size=population_size - elite_size, p=probs).tolist()
    except ValueError:
        selected = random.sample(population, population_size - elite_size)  # Wybierz losowo, jeśli błąd
    return selected


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
population = lib.create_population(population_size, start_point, end_point, grid_size, obstacles)
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
            child1, child2 = lib.crossover(parent1, parent2), lib.crossover(parent2, parent1)
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

