import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming

from solver import genetic_tsp
from utils import read_input

st.set_page_config(layout="wide")

"""
# Genetic TSP

Demo of genetic algorithm solver for the Traveling Salesman Problem.

Feel free to play with the parameters in the sidebar and see how they impact the
solution.

"""

with st.sidebar:
    select_dataset = st.selectbox(
        label="Select a dataset",
        options=("p01.in", "dj15.in", "dj38.in", "qa194.in"),
    )

    num_generations = st.number_input(
        "Number of generations", min_value=10, max_value=5000, step=10
    )

    population_size = st.number_input(
        "Population size", min_value=10, max_value=5000, step=10
    )

    hof_size = st.number_input(
        "Hall Of Fame size",
        min_value=1,
        max_value=population_size,
        help="Number of fittest individuals to carry over to the next generation. \
            Must be smaller than population size",
    )

    crossover_prob = st.number_input(
        "Crossover probability", min_value=0.0, max_value=1.0, step=0.01, value=0.8
    )

    mutation_prob = st.number_input(
        "Mutation probability", min_value=0.0, max_value=1.0, value=0.2
    )

    random_seed_checkbox = st.checkbox("Set a random seed?")

    if random_seed_checkbox:
        random_seed = st.number_input("Random seed", min_value=0, step=1, value=42)
        random.seed(random_seed)
        np.random.seed(random_seed)

col1, col2 = st.beta_columns(2)

col1.header("Best solution")
progress_bar = st.empty()
current_distance = st.empty()
plot = col1.empty()
done = st.empty()
final_distance = st.empty()
optimal_distance = st.empty()

col2.header("Distance over time")
df = pd.DataFrame({"Distance": []})
chart = col2.line_chart(df)


## Run the Genetic Algorithm
pop, stats, logbook, hof = genetic_tsp(
    select_dataset,
    num_generations,
    population_size,
    hof_size,
    crossover_prob,
    mutation_prob,
    chart,
    plot,
    progress_bar,
    current_distance,
)

progress_bar.empty()
current_distance.empty()

cities = read_input(f"data/{select_dataset}")

optimal_distances = {"p01.in": 284, "dj15.in": 3172, "dj38.in": 6656, "qa194.in": 9352}

done.write("**Done**!")
final_distance.write(f"**Final Distance:** {logbook[num_generations]['min']}")
optimal_distance.write(f"**Optimal Distance:** {optimal_distances[select_dataset]}")
