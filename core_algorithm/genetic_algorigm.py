from typing import List, Tuple

import numpy as np

from .target_function import rastrigin_function
from .data_types import Selection, Point, Population, Bounds


def initialize_population(pop_size: int, x_bounds: Point = Bounds, y_bounds: Point = Bounds) -> Population:
    """
    Initialize population randomly within given bounds.
    Each individual is a tuple (x, y).
    """
    population = []
    for _ in range(pop_size):
        x = np.random.uniform(*x_bounds)
        y = np.random.uniform(*y_bounds)
        population.append((x, y))
    return population


def fitness(individual: Point) -> float:
    """
    We aim to MINIMIZE the Rastrigin function,
    so let's define fitness = -f(x,y) for convenience
    (because standard GAs often maximize fitness).
    """
    x, y = individual
    return -rastrigin_function(x, y)


def selection(population: Population, scores: List[float], method: Selection = Selection.ROULETTE):
    """
    Select an individual from the population based on the method.
    For simplicity, we demonstrate roulette-wheel selection.
    Other methods: tournament, rank, etc.
    """
    if method == Selection.ROULETTE:
        # Normalize scores to create a probability distribution
        total_fitness = sum(scores)
        if total_fitness == 0:
            # If total_fitness is 0, pick random
            return population[np.random.randint(len(population))]

        probabilities = [s / total_fitness for s in scores]
        # Perform roulette-wheel selection
        r = np.random.rand()
        cum_prob = 0.0
        for i, p in enumerate(probabilities):
            cum_prob += p
            if r < cum_prob:
                return population[i]

    elif method == Selection.TOURNAMENT:
        # Simple tournament with 3 participants (example)
        k = 3
        chosen = np.random.choice(len(population), k, replace=False)
        selected = max(chosen, key=lambda idx: scores[idx])
        return population[selected]

    else:
        # Default random
        return population[np.random.randint(len(population))]


def crossover(parent1: Point, parent2: Point, crossover_rate: float = 0.9) -> Tuple[Point, Point]:
    """
    Single-point crossover on 2D chromosome (x,y).
    You could also do two-point or uniform crossover.
    """
    if np.random.rand() < crossover_rate:
        # Example: crossover on x or y with 50% probability
        if np.random.rand() < 0.5:
            # Swap x
            child1 = (parent2[0], parent1[1])
            child2 = (parent1[0], parent2[1])
        else:
            # Swap y
            child1 = (parent1[0], parent2[1])
            child2 = (parent2[0], parent1[1])
        return child1, child2
    else:
        # No crossover -> just copy parents
        return parent1, parent2


def mutate(
        individual: Point,
        mutation_rate: float = 0.1,
        x_bounds: Point = Bounds,
        y_bounds: Point = Bounds
) -> Point:
    """
    Mutate one of the genes (x or y) with some probability.
    """
    x, y = individual
    if np.random.rand() < mutation_rate:
        # Decide whether to mutate x or y
        if np.random.rand() < 0.5:
            x = np.random.uniform(*x_bounds)
        else:
            y = np.random.uniform(*y_bounds)
    return x, y


# def run(
#         pop_size: int = 50,
#         epochs: int = 20,
#         selection_method: Selection = Selection.ROULETTE,
#         crossover_rate: float = 0.9,
#         mutation_rate: float = 0.1
# ) -> Tuple[List[Point], List[float]]:
#     """
#     Main GA loop. Returns:
#         - best_individuals_per_epoch: list of (best_x, best_y) per epoch
#         - best_scores_per_epoch: list of best scores per epoch
#     """
#
#     # 1) Initialize population
#     population = initialize_population(pop_size)
#
#     best_individuals_per_epoch = []
#     best_scores_per_epoch = []
#
#     # 2) Start GA loop
#     for epoch in range(epochs):
#         # Evaluate fitness for each individual
#         scores = [fitness(ind) for ind in population]
#
#         # Track the best individual
#         best_score_epoch = max(scores)
#         best_ind_epoch = population[np.argmax(scores)]
#
#         best_individuals_per_epoch.append(best_ind_epoch)
#         best_scores_per_epoch.append(best_score_epoch)
#
#         # 3) Create new population
#         new_population = []
#         while len(new_population) < pop_size:
#             # Selection
#             parent1 = selection(population, scores, method=selection_method)
#             parent2 = selection(population, scores, method=selection_method)
#
#             # Crossover
#             child1, child2 = crossover(parent1, parent2, crossover_rate)
#
#             # Mutation
#             child1 = mutate(child1, mutation_rate=mutation_rate)
#             child2 = mutate(child2, mutation_rate=mutation_rate)
#
#             new_population.extend([child1, child2])
#
#         # Replace old population
#         population = new_population[:pop_size]
#
#     return best_individuals_per_epoch, best_scores_per_epoch