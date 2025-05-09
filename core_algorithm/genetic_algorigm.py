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


def negate_bit(bit: str):
    if bit == '0':
        return '1'
    else:
        return '0'


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


def selection(population: Population, scores: List[float], method: SelectionBox) -> Population:
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
    

def _one_point_cross(parent1: Point, parent2: Point):
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

def _two_points_cross(parent1: Point, parent2: Point):
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

def _homogeneous_cross(parent1: Point, parent2: Point, crossover_rate):
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

def _grain_cross(parent1, parent2):

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
    
def _arthmetic_cross(parent1, parent2):
    alpha = np.random.rand()
    if alpha == 0:
        alpha += 1e-10

    child1 = ((alpha * parent1[0] + (1 - alpha) * parent2[0]), (alpha * parent1[1] + (1 - alpha) * parent2[1]))
    child2 = ((alpha * parent2[0] + (1 - alpha) * parent1[0]), (alpha * parent2[1] + (1 - alpha) * parent1[1]))

    return child1, child2

def _type_alpha_cross(parent1, parent2, alpha):
    d1 = abs(parent1[0] - parent2[0])
    d2 = abs(parent1[1] - parent2[1])

    min_x = min([parent1[0], parent2[0]]) - alpha * d1
    max_x = max([parent1[0], parent2[0]]) + alpha * d1
    min_y = min([parent1[1], parent2[1]]) - alpha * d2
    max_y = max([parent1[1], parent2[1]]) + alpha * d2

    x1_new = np.random.uniform(min_x, max_x)
    x2_new = np.random.uniform(min_x, max_x)
    y1_new = np.random.uniform(min_y, max_y)
    y2_new = np.random.uniform(min_y, max_y)

    child1 = (x1_new, y1_new)
    child2 = (x2_new, y2_new)

    return child1, child2

def _type_alpha_and_beta_cross(parent1, parent2, alpha, beta):
    d1 = abs(parent1[0] - parent2[0])
    d2 = abs(parent1[1] - parent2[1])

    min_x = min([parent1[0], parent2[0]]) - alpha * d1
    max_x = max([parent1[0], parent2[0]]) + beta * d1
    min_y = min([parent1[1], parent2[1]]) - alpha * d2
    max_y = max([parent1[1], parent2[1]]) + beta * d2

    x1_new = np.random.uniform(min_x, max_x)
    x2_new = np.random.uniform(min_x, max_x)
    y1_new = np.random.uniform(min_y, max_y)
    y2_new = np.random.uniform(min_y, max_y)

    child1 = (x1_new, y1_new)
    child2 = (x2_new, y2_new)

    return child1, child2

def _average_cross(parent1, parent2):
    x_new = (parent1[0] + parent2[0]) / 2
    y_new = (parent1[1] + parent2[1]) / 2

    child = (x_new, y_new)

    return child

def _linear_cross(parent1, parent2, func):
    child_z = (0.5*(parent1[0] + parent2[0]), 0.5*(parent1[1] + parent2[1]))
    child_v = ((1.5*parent1[0]) - (0.5*parent2[0]), (1.5*parent1[1] - 0.5*parent2[1]))
    child_w = ((-0.5*parent1[0]) + (1.5*parent2[0]), (-0.5*parent1[1]) + (1.5*parent2[1]))
    children = [child_z, child_v, child_w] 

    score_z = fitness(func, child_z)
    score_v = fitness(func, child_v)
    score_w = fitness(func, child_w)
    scores = [score_z, score_v, score_w]

    scored_children = list(zip(children, scores))
    scored_children.sort(key=lambda scored_children: scored_children[1], reverse=True)

    child1 = scored_children[0][0]
    child2 = scored_children[1][0]

    return child1, child2

def crossover(parent1: Point, parent2: Point, crossover_rate: float, method: CrossMethodBox, alpha: float, beta: float, func):
    """
    Single-point crossover on 2D chromosome (x,y).
    You could also do two-point or uniform crossover.
    """
    if method == CrossMethodBox.ONEPOINT:
        if np.random.rand() < crossover_rate:
    # Example: crossover on x or y with 50% probability
            child1, child2 = _one_point_cross(parent1, parent2)

            return child1, child2
        else:
        # No crossover -> just copy parents
            return parent1, parent2
        
    elif method == CrossMethodBox.TWOPOINTS:
        if np.random.rand() < crossover_rate:
            child1, child2 = _two_points_cross(parent1, parent2)

            return child1, child2
        else:
            return parent1, parent2
    elif method == CrossMethodBox.HOMOGENEOUS:
          child1, child2 = _homogeneous_cross(parent1, parent2, crossover_rate)

          return child1, child2
    elif method == CrossMethodBox.GRAIN:
        if np.random.rand() < crossover_rate:
            child1 = _grain_cross(parent1, parent2)
           
            return child1
        else:
            return parent1
    elif method == CrossMethodBox.ARTHMETIC:
        if np.random.rand() < crossover_rate:
            child1, child2 = _arthmetic_cross(parent1, parent2)

            return child1, child2
        else:
            return parent1, parent2
    elif method == CrossMethodBox.TYPEALPHAMIX:
        if np.random.rand() < crossover_rate:
            child1, child2 = _type_alpha_cross(parent1, parent2, alpha)

            return child1, child2
        else:
            return parent1, parent2
    elif method == CrossMethodBox.TYPEALPHABETAMIX:
        if np.random.rand() < crossover_rate:
            child1, child2 = _type_alpha_and_beta_cross(parent1, parent2, alpha, beta)

            return child1, child2
        else:
            return parent1, parent2
    elif method == CrossMethodBox.AVERAGE:
        if np.random.rand() < crossover_rate:
            child1 = _average_cross(parent1, parent2)

            return child1
        else:
            return parent1
    elif method == CrossMethodBox.LINEAR:
        if np.random.rand() < crossover_rate:
            child1, child2 = _linear_cross(parent1, parent2, func)

            return child1, child2
        else:
            return parent1, parent2
    else:
        raise ValueError(f"Unknown crossover method: {method}")
        

def _edge_mutation(point: Point):
    point_bin = list(point_to_bin(point))

    if np.random.rand() < 0.5:
        point_bin[0] = negate_bit(point_bin[0])
    else:
        point_bin[-1] = negate_bit(point_bin[-1])
    
    point_bin = "".join(point_bin)

    mutated = bin_to_point(point_bin)

    return mutated

def _point_mutation(point: Point):
    while(True):
        point_bin = list(point_to_bin(point))
        a = random.sample(range(len(point_bin)), 1)[0]
        point_bin[a] = negate_bit(point_bin[a])
        point_bin = "".join(point_bin)
        mutated = bin_to_point(point_bin)

        if math.isnan(mutated[0]) or math.isnan(mutated[1]):
            continue

        return mutated

def _two_points_mutation(point: Point):
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
    
def _inversion_mutation(point: Point):
    point_bin = point_to_bin(point)
    while(True):
        a, b = sorted(random.sample(range(len(point_bin)), 2))
        point_bin = point_bin[:a] + point_bin[a:b][::-1] + point_bin[b:]
        mutated = bin_to_point(point_bin)

        if math.isnan(mutated[0]) or math.isnan(mutated[1]):
            continue

        return mutated

def _isosceles_mutation(point: Point, minimum: float, maximum: float):
    index = random.choice([0, 1])

    x, y = point
    new_value = np.random.uniform(minimum, maximum)

    if index == 0:
        x = new_value
    else:
        y = new_value

    mutated = (x, y)

    return mutated

def _gaussian_mutation(point: Point, loc: float, scale: float):
    x, y = point
    mutation_x = np.random.normal(loc=loc, scale=scale)
    mutation_y = np.random.normal(loc=loc, scale=scale)

    x += mutation_x
    y += mutation_y

    mutated = (x, y)

    return mutated

def mutate(
        individual: Point,
        method: MutationMethodBox,
        mutation_rate: float,
        minimum: float,
        maximum: float,
        loc: float,
        scale: float
) -> Point:
    """
    Mutate one of the genes (x or y) with some probability.
    """
    if np.random.rand() < mutation_rate:
        if method == MutationMethodBox.EDGE:
            x, y = _edge_mutation(individual)
        
        elif method == MutationMethodBox.ONEPOINT:
            x, y = _point_mutation(individual)
        
        elif method == MutationMethodBox.TWOPOINTS:
            x, y = _two_points_mutation(individual)
        
        elif method == MutationMethodBox.INVERSION:
            x, y = _inversion_mutation(individual)
        
        elif method == MutationMethodBox.ISOSCELES:
            x, y = _isosceles_mutation(individual, minimum, maximum)

        elif method == MutationMethodBox.GAUSSIAN:
            x, y = _gaussian_mutation(individual, loc, scale)
            
        else:
            raise ValueError(f"Unknown mutation method: {method}")
        
    else:
        x, y = individual
        
    return x, y
    