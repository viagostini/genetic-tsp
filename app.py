import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from solver import genetic_tsp
from utils import read_input

"""
# Genetic TSP

Demo of genetic algorithm solver for the Traveling Salesman Problem.

Feel free to play with the parameters in the sidebar and see how they impact the
solution.

"""


select_dataset = st.sidebar.selectbox(
    label="Select a dataset",
    options=("p01.in", "dj15.in", "dj38.in", "uy734.in"),
)

num_generations = st.sidebar.number_input(
    "Number of generations", min_value=10, max_value=1000, step=10
)

population_size = st.sidebar.number_input(
    "Population size", min_value=10, max_value=1000, step=10
)

hof_size = st.sidebar.number_input(
    "Hall Of Fame size",
    min_value=1,
    max_value=population_size,
    help="Number of fittest individuals to carry over to the next generation. \
        Must be smaller than population size",
)

crossover_prob = st.sidebar.number_input(
    "Crossover probability", min_value=0.0, max_value=1.0, step=0.01, value=0.8
)

mutation_prob = st.sidebar.number_input(
    "Mutation probability", min_value=0.0, max_value=1.0, value=0.2
)

random_seed_checkbox = st.sidebar.checkbox("Set a random seed?")


if random_seed_checkbox:
    random_seed = st.sidebar.number_input("Random seed", min_value=0, step=1, value=42)
    random.seed(random_seed)
    np.random.seed(random_seed)

## Run the Genetic Algorithm
pop, stats, logbook, hof = genetic_tsp(
    select_dataset,
    num_generations,
    population_size,
    hof_size,
    crossover_prob,
    mutation_prob,
)

cities = read_input(f"data/{select_dataset}")

solution = hof[0].tolist()
solution.append(solution[0])

fig, ax = plt.subplots()

ax.plot(
    [cities[i].x for i in solution],
    [cities[i].y for i in solution],
    "-o",
)
ax.plot(cities[0].x, cities[0].y, "r*")

st.pyplot(fig)

st.markdown(f"Final Distance: {logbook[num_generations]['min']}")
