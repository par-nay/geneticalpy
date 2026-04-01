# GeneticAlPy
### Genetic Algorithms with Python

Yet another package for lightweight applications of GA in Python.

This package provides utilities for implementation of Genetic Algorithm (Holland [1962](https://dl.acm.org/doi/10.1145/321127.321128)) for multivariate, multimodal optimization problems. Check out an introduction from a biological perspective in my paper, Nayak & Saha ([2022](https://academic.oup.com/mnras/article/510/2/2173/6462919))

By default, the binary representation is chosen as the genotype. The package provides roulette wheel, stochastic universal sampling and rank selection for choosing parents at each generation, uniform crossover, and bit-flip mutation. Elitist and liberal strategies may also be employed optionally.

## Quickstart 
Install the package with `pip`
```
pip install geneticalpy
```

Imports
```Python
from geneticalpy import genetical, examples
import numpy as np
import matplotlib.pyplot as plt
```

Define a fitness function from your cost function
```Python
cost = examples.ackley # replace this with your objective

def fitness(x): # to be maximized, by definition
    return 1/(1 + cost(x)) # this also depends loosely on your objective
```

Set initial GA hyperparameters
```Python
decimals   = 4
n_var      = 2
var_ranges = np.array([[-10,]*n_var,[10,]*n_var])
n_bits_segment    = len(format(int(max(var_ranges[1] - var_ranges[0])*10**decimals), 'b'))
n_bits_chromosome = n_bits_segment * n_var
offset     = -var_ranges[0]
init_popsize = 400
```

Initialize GA
```Python
PopGen = genetical.PopGenetics(
    fitness_func = fitness,
    n_var = n_var,
    decimal_acc = decimals,
    n_bits_chromosome = n_bits_chromosome,
)
pop_bin = PopGen.initialize_population(
    popsize = init_popsize, 
    var_ranges = var_ranges, 
    return_genotype = True,
    offset = offset,
)
```

Run evolution
```Python
evol_rec = PopGen.evolve(
    pop_bin, 
    n_gen    = 50, 
    n_pairs  = 800,
    elitist  = True,
    n_elites = 3,
    liberal  = False,
    n_runts  = 0,
    switch_selection = 5,
    prob_mut = 0.02, 
    prune    = True,
    pruning_cutoff = 800,
    verbose  = True,
    n_workers = 1,
)
```

Look at the results
```Python
print(f"Best solution is {evol_rec['fittest_individual']} with fitness {evol_rec['best_overall_fitness']}.")

plt.figure(figsize = (5,3))
gens = np.arange(len(evol_rec['best_fitness_per_generation']))+1
plt.plot(gens, evol_rec['best_fitness_per_generation'], color = 'green', label = 'Best')
plt.plot(gens, evol_rec['mean_fitness_per_generation'], color = 'blue', label = 'Mean')
plt.plot(gens, evol_rec['median_fitness_per_generation'], color = 'indigo', label = 'Median')
plt.legend()

plt.xlim(gens[0], gens[-1])
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
```


## Tutorial
Check out the notebook at `tutorial/tutorial.ipynb` for a lightning tutorial of GeneticAlPy!

## Get in touch
Drop me an email at [parth3e8@gmail.com](mailto:parth3e8@gmail.com) in case of any questions or to request more functionality!