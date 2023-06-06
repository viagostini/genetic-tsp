import random

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_solver import genetic_tsp
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
        options=("p01.in", "dj15.in", "dj38.in", "att48.in", "qa194.in"),
    )

    num_generations = st.number_input(
        "Number of generations", min_value=10, max_value=5000, step=10
    )

    population_size = st.number_input(
        "Population size", min_value=10, max_value=5000, step=10
    )

    mutation_prob = st.number_input(
        "Mutation probability", min_value=0.0, max_value=1.0, value=0.1
    )

    random_seed_checkbox = st.checkbox("Set a random seed?")

    if random_seed_checkbox:
        random_seed = st.number_input("Random seed", min_value=0, step=1, value=42)
        random.seed(random_seed)
        np.random.seed(random_seed)

col1, col2 = st.columns(2)

col1.header("Best solution")
progress_bar = st.empty()
current_distance = st.empty()
plot = col1.empty()
done = st.empty()
final_distance = st.empty()

optimal_distances = {
    "p01.in": 284,
    "dj15.in": 3172,
    "dj38.in": 6656,
    "att48.in": 33523,
    "qa194.in": 9352,
}
optimal_distance = st.write(
    f"**Optimal Distance:** {optimal_distances[select_dataset]}"
)

col2.header("Distance over time")
df = pd.DataFrame({"Distance": []})
chart = col2.empty()


## Run the Genetic Algorithm
best_solution, best_distance = genetic_tsp(
    select_dataset,
    num_generations,
    population_size,
    mutation_prob,
    chart,
    plot,
    progress_bar,
    current_distance,
)

progress_bar.empty()
current_distance.empty()

cities = read_input(f"data/{select_dataset}")


done.write("**Done**!")
final_distance.write(f"**Final Distance:** {best_distance}")
