import random
import numpy as np
import matplotlib.pyplot as plt

from lib import create_population
from lib import mutate
from lib import crossover
from lib import selection

# Parametry
population_size = 10      # Liczba osobników w populacji
mutation_rate = 0.5        # Prawdopodobieństwo mutacji
num_generations = 100     # Liczba generacji
elite_size = 5             # Liczba elitarnych osobników, które przechodzą do kolejnej generacji
grid_size = (50, 50)       # Rozmiar mapy (np. 20x20)
start_point = (0, 0)       # Punkt startowy
end_point = (30, 40)       # Punkt końcowy

# Przykładowe przeszkody
#obstacles = {(5, 5), (7, 6),  (6, 4), (10, 10), (17, 18)}
obstacles = {
                (5,6),(5,7),(5,8),(5,9),(5,10)
                ,(6,6),(6,7),(6,8),(6,9),(6,10),
                *((i,11) for i in range(5,8)),
                *((23,i) for i in range(20,42)),


            }

# Funkcja oceny ścieżki
def fitness(path):
    distance = len(path)
    collisions = sum(1 for p in path if p in obstacles)
    return distance + collisions * 100  # Kara za kolizje

# Główna pętla algorytmu EGA
population = create_population(obstacles=obstacles, population_size=population_size, grid_size=grid_size, end_point=end_point, start_point=start_point)
for generation in range(num_generations):

    # Sortowanie populacji według funkcji oceny
    population = sorted(population, key=fitness)
    
    # Elitaryzm: zachowanie najlepszych osobników
    new_population = population[:elite_size]
    
    # Tworzenie nowej generacji poprzez selekcję, krzyżowanie i mutację
    selected = selection(population, elite_size, fitness, population_size)

    for i in range(0, len(selected), 2):
        if i + 1 < len(selected):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            new_population.append(mutate(child1, obstacles, mutation_rate, grid_size))
            new_population.append(mutate(child2, obstacles, mutation_rate, grid_size))
    
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

