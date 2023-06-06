from src.individual import random_individual

def test_can_create_random_individual():
    num_genes = 5
    
    individual = random_individual(num_genes=num_genes)

    assert sorted(individual.genes) == list(range(num_genes)) 