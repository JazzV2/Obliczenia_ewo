import math
from typing import List, Tuple
import streamlit as st

import numpy as np
import bitstring
import random

from .target_function import TargetFunction
from .data_types import MutationMethodBox, SelectionBox, Point, Population, Bounds, CrossMethodBox

def point_to_bin(point: Point):
    x, y = point
    x_bin = bitstring.BitArray(float=x, length=32).bin
    y_bin = bitstring.BitArray(float=y, length=32).bin

    pointBin = x_bin + y_bin

    return pointBin

def bin_to_point(pointBin: str):
    x_bin, y_bin = pointBin[:32], pointBin[32:]

    x = bitstring.BitArray(bin=x_bin).float
    y = bitstring.BitArray(bin=y_bin).float

    return (x, y)

def one_point_cross(parent1: Point, parent2: Point):
    parent1_bin = point_to_bin(parent1)
    parent2_bin = point_to_bin(parent2)

    while(True):
        a = random.sample(range(1, len(parent1_bin) - 1), 1)[0]

        children1_bin = parent1_bin[:a] + parent2_bin[a:]
        children2_bin = parent2_bin[:a] + parent1_bin[a:]

        children1 = bin_to_point(children1_bin)
        children2 = bin_to_point(children2_bin)

        if math.isnan(children1[0]) or math.isnan(children1[1]) or math.isnan(children2[0]) or math.isnan(children2[1]):
            continue

        return children1, children2

def two_points_cross(parent1: Point, parent2: Point):
    parent1_bin = point_to_bin(parent1)
    parent2_bin = point_to_bin(parent2)

    while(True):
        a, b = sorted(random.sample(range(1, len(parent1_bin) - 1), 2))

        children1_bin = parent1_bin[:a] + parent2_bin[a:b] + parent1_bin[b:]
        children2_bin = parent2_bin[:a] + parent1_bin[a:b] + parent2_bin[b:]

        children1 = bin_to_point(children1_bin)
        children2 = bin_to_point(children2_bin)

        if math.isnan(children1[0]) or math.isnan(children1[1]) or math.isnan(children2[0]) or math.isnan(children2[1]):
            continue

        return children1, children2

def homogeneous_cross(parent1: Point, parent2: Point, crossover_rate):
    parent1_bin = point_to_bin(parent1)
    parent2_bin = point_to_bin(parent2)

    while(True):
        parent1_bin = list(parent1_bin)
        parent2_bin = list(parent2_bin)

        for i in range(len(parent1_bin)):
            if np.random.rand() < crossover_rate:
                bit = parent1_bin[i]
                parent1_bin[i] = parent2_bin[i]
                parent2_bin[i] = bit
        
        parent1_bin = "".join(parent1_bin)
        parent2_bin = "".join(parent2_bin)
        
        children1 = bin_to_point(parent1_bin)
        children2 = bin_to_point(parent1_bin)

        if math.isnan(children1[0]) or math.isnan(children1[1]) or math.isnan(children2[0]) or math.isnan(children2[1]):
            continue

        return children1, children2

def grain_cross(parent1, parent2):

    while(True):
        parent1_bin = list(point_to_bin(parent1))
        parent2_bin = list(point_to_bin(parent2))

        for i in range(len(parent1_bin)):
            if np.random.rand() >= 0.5:
                parent1_bin[i] = parent2_bin[i]
            
        parent1_bin = "".join(parent1_bin)
        children1 = bin_to_point(parent1_bin)

        if math.isnan(children1[0]) or math.isnan(children1[1]):
            continue
        
        return children1

def negate_bit(bit: str):
    if bit == '0':
        return '1'
    else:
        return '0'
    
def edge_mutation(point: Point):
    point_bin = list(point_to_bin(point))

    if np.random.rand() < 0.5:
        point_bin[0] = negate_bit(point_bin[0])
    else:
        point_bin[-1] = negate_bit(point_bin[-1])
    
    point_bin = "".join(point_bin)

    mutated = bin_to_point(point_bin)

    return mutated

def point_mutation(point: Point):
    while(True):
        point_bin = list(point_to_bin(point))
        a = random.sample(range(len(point_bin)), 1)[0]
        point_bin[a] = negate_bit(point_bin[a])
        point_bin = "".join(point_bin)
        mutated = bin_to_point(point_bin)

        if math.isnan(mutated[0]) or math.isnan(mutated[1]):
            continue

        return mutated

def two_points_mutation(point: Point):
    while(True):
        point_bin = list(point_to_bin(point))
        a, b = random.sample(range(len(point_bin)), 2)
        point_bin[a] = negate_bit(point_bin[a])
        point_bin[b] = negate_bit(point_bin[b])
        point_bin = "".join(point_bin)
        mutated = bin_to_point(point_bin)

        if math.isnan(mutated[0]) or math.isnan(mutated[1]):
            continue

        return mutated
    
def inversion_mutation(point: Point):
    point_bin = point_to_bin(point)
    while(True):
        a, b = sorted(random.sample(range(len(point_bin)), 2))
        point_bin = point_bin[:a] + point_bin[a:b][::-1] + point_bin[b:]
        mutated = bin_to_point(point_bin)

        if math.isnan(mutated[0]) or math.isnan(mutated[1]):
            continue

        return mutated

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


def fitness(function: TargetFunction, individual: Point) -> float:
    """
    We aim to MINIMIZE the target function,
    so let's define fitness = -f(x,y) for convenience
    (because standard GAs often maximize fitness).
    """
    x, y = individual
    return -function(x, y)


def selection(population: Population, scores: List[float], method: SelectionBox = str(SelectionBox.ROULETTE)):
    """
    Select an individual from the population based on the method.
    For simplicity, we demonstrate roulette-wheel selection.
    Other methods: tournament, rank, etc.
    """
    if method == str(SelectionBox.ROULETTE):
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

    elif method == str(SelectionBox.TOURNAMENT):
        # Simple tournament with 3 participants (example)
        k = 3
        chosen = np.random.choice(len(population), k, replace=False)
        selected = max(chosen, key=lambda idx: scores[idx])
        return population[selected]
    
    elif method == str(SelectionBox.THEBEST):
        population_indexes = np.arange(0, len(population))
        selected = max(population_indexes, key=lambda idx: scores[idx])
        return population[selected]
        
    else:
        # Default random
        return population[np.random.randint(len(population))]


def crossover(parent1: Point, parent2: Point, crossover_rate: float = 0.9, method: CrossMethodBox = str(CrossMethodBox.ONEPOINT)) -> Tuple[Point, Point]:
    """
    Single-point crossover on 2D chromosome (x,y).
    You could also do two-point or uniform crossover.
    """
    
    if method == str(CrossMethodBox.ONEPOINT):
        if np.random.rand() < crossover_rate:
    # Example: crossover on x or y with 50% probability
            child1, child2 = one_point_cross(parent1, parent2)

            return child1, child2
        else:
        # No crossover -> just copy parents
            return parent1, parent2
        
    elif method == str(CrossMethodBox.TWOPOINTS):
        if np.random.rand() < crossover_rate:
            child1, child2 = two_points_cross(parent1, parent2)

            return child1, child2
        else:
            return parent1, parent2
    elif method == str(CrossMethodBox.HOMOGENEOUS):
          child1, child2 = homogeneous_cross(parent1, parent2, crossover_rate)

          return child1, child2
    elif method == str(CrossMethodBox.GRAIN):
        if np.random.rand() < crossover_rate:
            child1 = grain_cross(parent1, parent2)

            return child1
        else:
            return parent1


def mutate(
        individual: Point,
        method,
        mutation_rate: float = 0.1
) -> Point:
    """
    Mutate one of the genes (x or y) with some probability.
    """
    if np.random.rand() < mutation_rate:
        if (method == str(MutationMethodBox.EDGE)):
            x, y = edge_mutation(individual)
            return x, y
        
        elif (method == str(MutationMethodBox.ONEPOINT)):
            x, y = point_mutation(individual)
            return x, y
        
        elif (method == str(MutationMethodBox.TWOPOINTS)):
            x, y = two_points_mutation(individual)
            return x, y
        elif (method == str(MutationMethodBox.INVERSION)):
            x, y = inversion_mutation(individual)
            return x, y
    else:
        x, y = individual

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