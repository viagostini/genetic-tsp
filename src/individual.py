from numpy.random import permutation

class Individual:
    def __init__(self, genes: list[int]):
        self.genes = genes

def random_individual(num_genes: int) -> Individual:
    return Individual(genes=permutation(range(num_genes)))