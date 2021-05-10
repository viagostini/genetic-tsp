import array
import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from deap import algorithms, base, creator, tools

from utils import read_input


def genetic_tsp(
    dataset_name: str,
    num_generations: int,
    population_size: int,
    hof_size: int,
    crossover_prob: float,
    mutation_prob: float,
    chart,
    plot,
    progress_bar,
    current_distance,
):
    cities = read_input(f"data/{dataset_name}")
    NUM_CITIES = len(cities)

    distance_map = [[city.distance_to(other) for city in cities] for other in cities]
    IND_SIZE = NUM_CITIES

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

    # Structure initializers
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=10)
    toolbox.register("evaluate", evalTSP, distance_map=distance_map)

    pop = toolbox.population(n=population_size)

    hof = tools.HallOfFame(hof_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, logbook = perform_evolution(
        cities,
        pop,
        toolbox,
        chart,
        plot,
        progress_bar,
        current_distance,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=num_generations,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    return pop, stats, logbook, hof


def evalTSP(individual, distance_map):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return (distance,)


def perform_evolution(
    cities,
    population,
    toolbox,
    chart,
    plot,
    progress_bar,
    current_distance,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    progress_bar.progress(0)

    current_distance.text("")

    solution = halloffame[0].tolist()
    solution.append(solution[0])

    fig, ax = plt.subplots()

    ax.plot(
        [cities[i].x for i in solution],
        [cities[i].y for i in solution],
        "-o",
    )
    ax.plot(cities[0].x, cities[0].y, "r*")

    plot.pyplot(fig)

    chart.line_chart()

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - len(halloffame))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        progress_bar.progress(int(gen / ngen * 100))
        current_distance.write(f"Current distance: {logbook[gen]['min']}")
        chart.add_rows({"Distance": [logbook[gen]["min"]]})

        solution = halloffame[0].tolist()
        solution.append(solution[0])
        ax.clear()
        ax.plot(
            [cities[i].x for i in solution],
            [cities[i].y for i in solution],
            "-o",
        )
        ax.plot(cities[0].x, cities[0].y, "r*")

        plot.pyplot(fig)
    progress_bar.empty()

    return population, logbook
