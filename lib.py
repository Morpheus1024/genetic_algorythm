import numpy as np
import random

def create_population(obstacles, population_size=100, grid_size = (50,50), end_point = (41,25), start_point = (1,1)):
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
            # next_steps = [
            #     (x + dx, y),
            #     (x, y + dy),
            #     (x + dx, y + dy),
            # ]
            # random.shuffle(next_steps)

            # for next_step in next_steps:
            #     next_step =(
            #         max(0, min(next_step[0], grid_size[0] - 1)),
            #         max(0, min(next_step[1], grid_size[1] - 1))
            #     )

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


def crossover(parent1, parent2):
    split = random.randint(1, min(len(parent1), len(parent2)) - 2)
    child = parent1[:split] + [p for p in parent2 if p not in parent1[:split]]
    return child

def mutate(path, obstacles, mutation_rate, grid_size):
    if random.random() < mutation_rate:
        idx = random.randint(1, len(path) - 2)
        x, y = path[idx]
        new_step = (x + random.choice([-1, 0, 1]), y + random.choice([-1, 0, 1]))
        new_step = (max(0, min(new_step[0], grid_size[0] - 1)),
                    max(0, min(new_step[1], grid_size[1] - 1)))
        if new_step not in obstacles and new_step not in path:
            path[idx] = new_step
    return path

def selection(population, elite_size, fitness, population_size):
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

