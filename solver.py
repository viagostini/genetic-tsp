import math
import random

import numpy as np

from utils import pairwise, random_interval, random_population, read_input

random.seed(42)
np.random.seed(42)


cities = None


def crossover(ind1, ind2):
    size = len(ind1)

    i, j = random_interval(ind1)

    c1 = ind1[i:j]
    c2 = ind2[i:j]

    for k in range(size):
        child_pos = (j + k) % size

        if ind2[child_pos] not in c1:
            c1.append(ind2[child_pos])

        if ind1[child_pos] not in c2:
            c2.append(ind1[child_pos])

    c1 = c1[-i:] + c1[:-i]
    c2 = c2[-i:] + c2[:-i]

    return c1, c2


def evaluate_fitness(individual):
    dist = 0
    for x, y in pairwise(individual):
        dist += math.dist(cities[y], cities[x])
    return 1 / dist


def mutate(individual, prob):
    for i in range(len(individual)):
        if random.random() < prob:
            i, j = random_interval(individual)
            individual[i], individual[j] = individual[j], individual[i]
    return individual


def breed(population, mutation_prob):
    offspring = []

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            child1, child2 = crossover(population[i], population[j])
            child1 = mutate(child1, mutation_prob)
            child2 = mutate(child2, mutation_prob)
            offspring.extend([child1, child2])

    return population + offspring


def rank_selection(population, num_selected):
    pop_by_fitness = sorted(
        population, key=lambda ind: evaluate_fitness(ind), reverse=True
    )
    return pop_by_fitness[:num_selected]


def genetic_tsp(dataset_name, num_generations, population_size, mutation_prob):
    global cities
    cities = read_input(f"data/{dataset_name}")

    population = random_population(population_size, len(cities))

    for gen in range(num_generations):
        population_with_offspring = breed(population, mutation_prob)
        population = rank_selection(population_with_offspring, population_size)

        pop_fitness = [evaluate_fitness(individual) for individual in population]
        best_solution = population[np.argmax(pop_fitness)]
        best_distance = 1 / evaluate_fitness(best_solution)

        if gen % 10 == 0:
            print(f"[Gen {gen}] - Current Distance: {best_distance}")

    return best_solution, best_distance


best_solution, best_distance = genetic_tsp(
    dataset_name="dj15.in",
    num_generations=30,
    population_size=20,
    mutation_prob=0.1,
)

print(f"Best distance: {best_distance}")
