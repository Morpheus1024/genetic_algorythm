import random
import numpy as np

# Funkcja tworzenia początkowej populacji
def create_population(population_size: int, start_point:tuple[int,int], end_point:tuple[int,int], grid_size: tuple[int,int], obstacles: set[tuple[int,int]]) -> list:
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

# Operator krzyżowania
def crossover(parent1, parent2):
    split = random.randint(1, min(len(parent1), len(parent2)) - 2)
    child = parent1[:split] + [p for p in parent2 if p not in parent1[:split]]
    return child


def read_obstacles_from_file(file_path: str) ->  set[tuple[int,int]]:
    obstacles = set()
    with open(file_path, 'r') as file:
        for line in file:
            line = line[1:-2] if line [-1] == '\n' else line[1:-1] # usuwa nawiasy oraz znak entera
            x, y = map(int, line.strip().split(','))
            obstacles.add((x, y))
    return obstacles