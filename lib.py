import numpy as np
import random

def create_population(obstacles, population_size=100, grid_size = (50,50), end_point = (49,49), start_point = (1,1)):
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
