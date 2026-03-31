#!/usr/bin/env python

import sys
import warnings
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def bin_str2arr(bin_str):
    """
    Convert a binary string (chromosome) to a numpy array of 0s and 1s.
    
    Parameters:
    -----------
    bin_s : str
    A string representing a binary number, e.g., '101011'.
        
    Returns:
    --------
    np.ndarray
    A numpy array of 0s and 1s corresponding to the input binary string.
    """
    l	= list(bin_str)
    return np.array(l, dtype = 'int')


def bin_arr2str(bin_arr):
    """
    Convert a numpy array of bits to a binary string (chromosome).

    Parameters:
    -----------
    bin_arr : np.ndarray
    A numpy array of 0s and 1s corresponding to the chromosome.

    Returns:
    --------
    bin_s : str
    The string representing the input binary vector, e.g., '101011'.
    """
    bin_str = ''.join(list(bin_arr.astype(int).astype(str)))
    return bin_str


def dec2bin(dec_arr, n_bits_chromosome, decimal_acc, offset = 0):
    """
    Convert a vector of decimal numbers to a concatenated binary string (chromosome representation).
    
    Parameters:
    -----------
    dec_arr : np.ndarray
    Input decimal vector
        
    n_bits_chromosome : int
    Desired size of the full chromosomes in bits
        
    decimal_acc : int
    Desired decimal accuracy (number of decimal places)
    
    offset : float [np.ndarray], optional
    Any offset (vector) to be added to the input decimal array before conversion. Defaults to 0.
        
    Returns:
    --------
    str
    The chromosome representation of the input decimal vector
    """
    n_var   = dec_arr.shape[-1]
    divsize = n_bits_chromosome // n_var
    dec_arr = dec_arr + offset
    # Vectorized: scale and round all at once
    int_arr = np.round(dec_arr * (10**decimal_acc)).astype(int)
    # Convert each to binary with padding and concatenate
    bin_strs = [format(num, f'0{divsize}b') for num in int_arr]
    bin_str = "".join(bin_strs)
    return bin_str


def bin2dec(bin_arr, n_bits_segment, decimal_acc, offset = 0):
    """
    Convert a chromosome vector of bits (0s and 1s) to a decimal vector (phenotype representation).
    
    Parameters:
    -----------
    bin_arr : np.ndarray
    Input chromosome vector (could have been created using `bin_str2arr`)
        
    n_bits_segment : int
    Size of each independent variable of the vector in bits (also called a segment of the chromosome)
        
    decimal_acc : int
    Desired decimal accuracy (number of decimal places)
    
    offset : float [np.ndarray], optional
    Any offset (vector) to be subtracted from the decimal array after conversion. Defaults to 0.
        
    Returns:
    --------
    np.ndarray
    The phenotype representation (decimal vector) of the input chromosome (binary vector)
    """
    n 		= len(bin_arr)
    n_var 	= n // n_bits_segment
    dec_arr = np.zeros(n_var)
    # Vectorized: reshape into segments and convert each
    for i in range(n_var):
        segment_str = bin_arr2str(bin_arr[i*n_bits_segment:(i+1)*n_bits_segment])
        dec_arr[i] = int(segment_str, 2) / (10**decimal_acc)
    dec_arr = dec_arr - offset
    return dec_arr


def pop_dec2bin(pop_dec, n_bits_chromosome, decimal_acc, offset = 0):
    """
    Convert a population of decimal vectors to their genotype representation (binary vectors).

    Parameters:
    -----------
    pop_dec : np.ndarray
    A population in phenotype representation (array of shape `(popsize, n_var)` containing decimal vectors of individuals)

    n_bits_chromosome : int
    Desired size of a full chromosome in bits

    decimal_acc : int
    Desired decimal accuracy (number of decimal places)

    offset : float [np.ndarray], optional
    Any offset (vector) to be added to the input decimal arrays before conversion. Defaults to 0.

    Returns:
    --------
    str
    The genotype representation (binary vectors) of the input decimal population
    """
    popsize = pop_dec.shape[0]
    n_var = pop_dec.shape[1]
    divsize = n_bits_chromosome // n_var
    pop_bin = np.zeros((popsize, n_bits_chromosome), dtype=int)
    
    # Vectorized: convert all at once instead of looping through individuals
    pop_dec_offset = pop_dec + offset
    int_pop = np.round(pop_dec_offset * (10**decimal_acc)).astype(int)
    
    # Fill binary array efficiently
    for i in range(popsize):
        for j in range(n_var):
            bin_str = format(int_pop[i, j], f'0{divsize}b')
            pop_bin[i, j*divsize:(j+1)*divsize] = np.array([int(b) for b in bin_str])
    
    return pop_bin


def pop_bin2dec(pop_bin, n_bits_segment, decimal_acc, offset = 0):
    """
    Convert a population of binary vectors to their phenotype representation (decimal vectors).

    Parameters:
    -----------
    pop_bin : np.ndarray
    A population in genotype representation (array of shape `(popsize, n_bits_chromosome)` containing binary vectors of individuals)

    n_bits_segment : int
    Size of each independent variable (of an individual vector in the population) in bits (also called a segment of a chromosome)

    decimal_acc : int
    Desired decimal accuracy (number of decimal places)

    offset : float [np.ndarray], optional
    Any offset (vector) to be subtracted from the decimal arrays after conversion. Defaults to 0.

    Returns:
    --------
    np.ndarray
    The phenotype representation (decimal vector) of the input binary population
    """
    popsize = pop_bin.shape[0]
    n_var = pop_bin.shape[1] // n_bits_segment
    pop_dec = np.zeros((popsize, n_var))
    
    # Vectorized: convert all at once
    for i in range(popsize):
        for j in range(n_var):
            segment_str = bin_arr2str(pop_bin[i, j*n_bits_segment:(j+1)*n_bits_segment])
            pop_dec[i, j] = int(segment_str, 2) / (10**decimal_acc)
    
    pop_dec = pop_dec - offset
    return pop_dec


def crossover(
    mating_pool, 
    type = 'uniform',
    seed = 42,
    random_rng = None,
):
    """
    Perform crossover on the mating pool to produce offspring.

    Parameters:
    -----------
    mating_pool : np.ndarray
    A mating pool (pairs of individuals) of shape `(n_pairs, 2, n_bits_chromosome)`.

    type : str, optional
    Type of (numerical) crossover to perform. Currently only 'uniform' is implemented. Defaults to 'uniform'.

    seed : int, optional
    Pseudorandom seed for reproducibility of crossover. Defaults to 42. Overridden by `random_rng` if supplied.

    random_rng : np.random.Generator, optional
    Supplied if an instance of the random number generator exists that must be used for creating pseudorandom numbers for crossover. Overrides `seed`. Defaults to `None`.
    """
    offspring = []
    if random_rng is None:
        random_rng   = np.random.default_rng(seed)
    if type == 'uniform':
        for i in range(len(mating_pool)):
            g1, g2 = mating_pool[i]
            for j in range(len(g1)):
                r = random_rng.random()
                if r > 0.5:
                    tmp   = g1[j]
                    g1[j] = g2[j]
                    g2[j] = tmp
            offspring.append(g1)
            offspring.append(g2)
        offspring = np.array(offspring)
        return offspring
    else:
        raise ValueError("Crossover type not recognized. Currently only 'uniform' works.")
    

def mutate(
    pop, 
    prob_mut,
    seed = 42,
    random_rng = None,
):
    """
    Mutate a population by flipping bits with a given mutation probability.

    Parameters:
    -----------
    pop : np.ndarray
    A population in genotype representation (array of shape `(popsize, n_bits_chromosome)` containing binary vectors of individuals)

    prob_mut : float
    Probability of mutation for each bit (between 0 and 1)

    seed : int, optional
    Pseudorandom seed for reproducibility of mutation. Defaults to 42. Overridden by `random_rng` if supplied.

    random_rng : np.random.Generator, optional
    Supplied if an instance of the random number generator exists that must be used for creating pseudorandom numbers for mutation. Overrides `seed`. Defaults to `None`.
    """
    if prob_mut == 0.0:
        return pop
    elif prob_mut < 0.0 or prob_mut > 1.0:
        raise ValueError("Mutation probability must be between 0 and 1.")
    else:
        if random_rng is None:
            random_rng = np.random.default_rng(seed)
        for i in range(len(pop)):
            for j in range(len(pop[i])):
                r = random_rng.random()
                if r < prob_mut:
                    pop[i][j] = float(not(pop[i][j]))
        return pop


class PopGenetics:
    """ 
    A class encapsulating population genetics operations for a genetic algorithm, including population initialization, mating pool creation, breeding, and evolution over generations.
    """
    def __init__(
        self, 
        fitness_func, 
        n_var, 
        decimal_acc, 
        n_bits_chromosome,
        seed_popinit   = 42,
        seed_selection = 43,
        seed_crossover = 44,
        seed_mutation  = 45,
    ):
        """
        Initialize the population genetics class

        Parameters:
        -----------
        fitness_func : callable
        Function to evaluate the fitness of an individual or the entire population. Takes phenotype(s) as input.

        n_var : int
        Number of varibles in an individual vector -- the dimensionality of the problem to be solved

        decimal_acc : int
        Desired decimal accuracy (in decimal places)

        n_bits_chromosome : int
        Desired size of a full chromosome in bits

        seed_[popinit, selection, crossover, mutation] : int, optional
        Pseudorandom seed to initialize random number generators for creating the initial population, selecting pairs in mating pools, crossover, and mutation, respectively. Defaults to [42, 43, 44, 45].
        """
        self.fitness_func  = fitness_func
        self.n_var         = n_var
        self.decimal_acc   = decimal_acc
        self.n_bits_chromosome  = n_bits_chromosome
        self.n_bits_segment     = n_bits_chromosome // n_var
        self.rng_popinit   = np.random.default_rng(seed_popinit)
        self.rng_selection = np.random.default_rng(seed_selection)
        self.rng_crossover = np.random.default_rng(seed_crossover)
        self.rng_mutation  = np.random.default_rng(seed_mutation)
        self._gen = 0
        
    def initialize_population(
        self, 
        popsize, 
        var_ranges,
        dist = 'uniform', 
        return_genotype = False,
        offset = 0,
    ):
        """ 
        Initialize a population by stochastically sampling individuals in the required variable ranges

        Parameters:
        -----------
        popsize : int
        Desired number of individuals in the initial population (population size)

        var_ranges : np.ndarray
        Lower and upper boundaries (inclusive) of variables to sample the initial population within, array of shape `(2, n_var)`.

        dist : str, optional
        Prior probability distribution for sampling the initial population. Currently only 'uniform' is supported. Defaults to 'uniform'.

        return_genotype : bool, optional
        Whether to return the initial population in the genotype representation (binary vectors). Defaults to False.

        offset : float [np.ndarray], optional
        Any offset (vector) to be added to the decimal arrays before conversion to the binary genotype. Defaults to 0.

        Returns:
        -----------
        np.ndarray
        Initial population of shape `(popsize, n_var)` or `(popsize, n_bits_chromosome)` (if `return_genotype` is `True`)
        """
        self.offset = offset
        pop = []
        if dist == 'uniform':
            pop = self.rng_popinit.uniform(
                low  = var_ranges[0],
                high = var_ranges[1],
                size = (popsize, self.n_var)
            )
        else:
            raise ValueError("Distribution type not recognized. Currently only 'uniform' works.")
        if return_genotype:
            pop_bin = []
            for indiv in pop:
                indiv_bin = dec2bin(
                    dec_arr = indiv,
                    n_bits_chromosome = self.n_bits_chromosome,
                    decimal_acc = self.decimal_acc,
                    offset = offset,
                )
                pop_bin.append(
                    bin_str2arr(indiv_bin)
                )
            pop_bin = np.array(pop_bin)
            return pop_bin
        else:
            return pop
        

    def create_mating_pool(
        self,
        pop, 
        fitness_arr,
        n_pairs,
        selection_type = 'SUS', 
        rank_selection = False,
        sp_rank = 2.0,
    ):
        """
        Create a mating pool by selecting pairs of individuals from the given population based on their fitness (selection pressure).

        Parameters:
        -----------
        pop : np.ndarray
        Binary population array of shape `(popsize, n_bits_chromosome)` where `n_bits_chromosome` is the size of each chromosome in bits. 

        fitness_arr : np.ndarray
        1D array of fitness values corresponding to the individuals in the (breeding) population.

        n_pairs : int
        Number of pairs to be selected for mating. (Pairs are selected with replacement - one pair can produce multiple independent children.)

        selection_type : str, optional
        Type of parent selection method to use. Options are: 'RW' for Roulette Wheel selection and 'SUS' for Stochastic Universal Sampling. The latter is a variation of the former with N_dim (angularly) equidistant pointers on the roulette wheel, where N_dim is the dimensionality of a seletion (e.g., for a "pair", N_dim = 2). Defaults to 'SUS'.

        rank_selection : bool, optional
        If `True`, applies a rank-based selection pressure instead of a fitness-proportionate one. Defaults to `False`.

        sp_rank : float, optional
        If rank selection is applied, the selection pressure parameter of [...], 1 <= sp_rank <= 2, where 1 is no selection pressure (flat PDF) and 2 is the steepest linear selection pressure. Defaults to 2.

        Returns:
        --------
        np.ndarray
        An array of shape `(n_pairs, 2, n_bits_chromosome)` - the mating pool.

        Raises:
        -------
        ValueError
        If an unrecognized selection type is requested. Currently recognized types are 'RW' and 'SUS'.
        """

        # first sort the population according to fitness
        indsort 	 = np.argsort(fitness_arr)
        pop 		 = pop[indsort]
        fitness_arr  = fitness_arr[indsort]
        if rank_selection:
            ranks      = np.arange(len(pop), 0, -1)
            # probs      = 1./ ranks
            probs      = sp_rank - 2*(sp_rank - 1)*(ranks - 1)/(len(ranks)-1)
            probs      = probs / np.sum(probs) # normalization
            cuml_probs = np.array([sum(probs[0:k+1]) for k in range(0, len(pop))])
        else:
            probs      = fitness_arr / np.sum(fitness_arr) # normalization
            cuml_probs = np.array([sum(probs[0:k+1]) for k in range(0, len(pop))])
        
        pairs = []

        if selection_type.lower() == 'rw': 	# this is the simple roulette wheel selection type (for selecting individuals)
            for i in range(n_pairs):
                r1 	= self.rng_selection.random()
                r2 	= self.rng_selection.random()
                j 	= 0 
                k 	= 0
                while cuml_probs[j] < r1:
                    j 	= j + 1
                while cuml_probs[k] < r2:
                    k 	= k + 1
                if j == k:
                    if k == 0:
                        k = k + 1
                    else:
                        k = k - 1
                pair 	= [j, k]
                pairs.append(pop[pair])
        
        elif selection_type.lower() == 'sus': 	# this is the stochastic universal sampling
            for i in range(n_pairs):
                r1 	= self.rng_selection.random()*0.5
                r2 	= r1 + 0.5
                j 	= 0 
                k 	= 0
                while cuml_probs[j] < r1:
                    j 	= j + 1
                while cuml_probs[k] < r2:
                    k 	= k + 1
                if j == k:
                    if k == 0:
                        k = k + 1
                    else:
                        k = k - 1 
                pair 	= [j, k]
                pairs.append(pop[pair])

        else:
            raise ValueError("Selection type not recognized. Use 'RW' or 'SUS'.")
            
        return np.array(pairs)
    
    def breed(
        self,
        mating_pool,
        crossover_type = 'uniform',
        prob_mut = 0.0,
        prune = False,
        pruning_cutoff = None,
        return_fitness = False,
        n_workers = None,
    ):
        """
        Breed the mating pool (pairs) by performing crossover on the chromosomes to produce offspring and then mutating them. An optional elitist pruning of the offspring based on their fitness values can be applied.

        Parameters:
        -----------
        mating_pool: np.ndarray
        The mating pool created using `create_mating_pool`, of shape `(n_pairs, 2, n_bits_chromosome)`.

        crossover_type : str, optional
        Type of (numerical) crossover to perform. Currently only 'uniform' is supported. Defaults to 'uniform'.

        prob_mut : float
        Probability of mutation for each bit (between 0 and 1)

        prune : bool, optional
        Whether to apply an "elitist" pruning of the offspring based on their fitness values. If `True`, a pruning threshold must be supplied. Defaults to `False`.

        pruning_cutoff : int, optional
        If pruning, the number of fittest offspring to keep. Must be supplied if `prune` is `True`. Defaults to `None`.

        return_fitness : bool, optional
        Whether to return the fitness values of the offspring along with the offspring. Defaults to `False`.

        Returns:
        --------
        np.ndarray or (np.ndarray, np.ndarray)
        The offspring produced after crossover, mutation and any pruning if applied. Returns the fitness values of the offspring as well (second returned object) if `return_fitness` is `True`.

        Raises:
        -------
        ValueError
        If an unrecognized crossover type is supplied. Currently only 'uniform' is recognized.

        ValueError
        If pruning is requested but a pruning cutoff is not supplied.
        """
        offspring = []
        offspring = crossover(
            mating_pool = mating_pool,
            type = crossover_type,
            random_rng = self.rng_crossover,
        )
        offspring = mutate(
            pop = offspring,
            prob_mut = prob_mut,
            random_rng = self.rng_mutation,
        )
        if return_fitness:
            if n_workers is None:
                n_workers = cpu_count()
            offspring_dec = pop_bin2dec(
                pop_bin = offspring,
                n_bits_segment = self.n_bits_segment,
                decimal_acc = self.decimal_acc,
                offset = self.offset,
            )
            if n_workers > 1:
                with Pool(processes = n_workers) as pool:
                    fitness_offspring = np.array(pool.map(self.fitness_func, offspring_dec))
            else:
                fitness_offspring = self.fitness_func(offspring_dec)
        if prune:
            if pruning_cutoff is None:
                raise ValueError("If pruning, a pruning cutoff (in number of offspring to keep) must be supplied.")
            if not return_fitness:
                offspring_dec = pop_bin2dec(
                    pop_bin = offspring,
                    n_bits_segment = self.n_bits_segment,
                    decimal_acc = self.decimal_acc,
                    offset = self.offset,
                )
                if n_workers > 1:
                    with Pool(processes = n_workers) as pool:
                        fitness_offspring = np.array(pool.map(self.fitness_func, offspring_dec))
                else:
                    fitness_offspring = self.fitness_func(offspring_dec)
            indsort   = np.argsort(fitness_offspring)
            offspring = offspring[indsort][-pruning_cutoff:]
            fitness_offspring = fitness_offspring[indsort][-pruning_cutoff:]

        if return_fitness:
            return offspring, fitness_offspring
        else:
            return offspring
        
    def evolve(
        self, 
        pop, 
        n_gen, 
        n_pairs,
        selection_type = 'SUS',
        elitist = True,
        n_elites = 1,
        liberal = False,
        n_runts = 1,
        switch_selection  = None,
        sp_rank_selection = 2.0,
        crossover_type = 'uniform',
        prob_mut = 0.0,
        prune = False,
        pruning_cutoff = None,
        verbose = True,
        n_workers = None,
    ):
        """
        Evolve a population of individuals by natural selection for a fixed number of generations.

        Note: This method supports checkpointing. You can call it multiple times with increasing `n_gen` to continue evolution from where it left off, maintaining reproducibility and progress tracking. Upon subsequent calls, it is possible to supply new parameter values, except `pop` (any `pop` supplied would be completely ignored). `switch_selection` remains universal.

        Parameters:
        -----------
        pop : np.ndarray 
        Binary population array of shape `(popsize, n_bits_chromosome)` where `n_bits_chromosome` is the size of each chromosome in bits. 

        n_gen : int
        Number of generation to evolve the population for 

        n_pairs: int 
        Number of mating pairs to pick in each mating pool 

        selection_type : str, optional
        Type of parent selection method to use. Options are: 'RW' for Roulette Wheel selection and 'SUS' for Stochastic Universal Sampling. The latter is a variation of the former with N_dim (angularly) equidistant pointers on the roulette wheel, where N_dim is the dimensionality of a seletion (e.g., for a "pair", N_dim = 2). Defaults to 'SUS'. 

        elitist : bool, optional
        If True, the fittest individual(s) at each generation is (are) carried over directly to the next (see `n_elites`). Defaults to True.

        n_elites : int, optional 
        If an elitist strategy is adopted, the number of fittest individuals to carry over to the next. Defaults to 1.

        liberal : bool, optional
        If True, the least fit individual(s) at each generation is (are) carried over to the next (as an attempt to preserve genetic diversity). Recommended to always use in combination with `elitist`. Defaults to False.

        n_runts : int, optional
        If a liberal strategy is adopted, the number of least fit individuals to carry over to the next. Defaults to 1.

        switch_selection : int, optional 
        Optional switching generation for the selection type from fitness-proportionate to rank-based. Defaults to None (in which case no switching is applied).

        sp_rank_selection : float, optional
        If rank selection is applied, the selection pressure parameter of [...], 1 <= sp <= 2, where 1 is no selection pressure (flat PDF) and 2 is the steepest linear selection pressure. Defaults to 2.

        crossover_type : str, optional 
        Type of (numerical) crossover to perform. Currently only 'uniform' is supported. Defaults to 'uniform'. 

        prob_mut : float, optional 
        Probability of mutation for each bit (between 0 and 1)

        prune : bool, optional
        Whether to apply an "elitist" pruning of the offspring based on their fitness values. If `True`, a pruning threshold must be supplied. Defaults to `False`.

        pruning_cutoff : int, optional
        If pruning, the number of fittest offspring to keep. Must be supplied if `prune` is `True`. Defaults to `None`. 

        verbose : bool, optional
        Whether to show progress of the evolution as a progress bar. Defaults to True.

        Returns:
        -----------
        dict 
        The results of the evolution recorded in a dictionary with the following items:
        [key : type
        description]
        - 'fittest_individual' : np.ndarray 
        The fittest solution in decimal (phenotype) representation found by evolution 

        - 'best_overall_fitness' : float 
        The fitness value of the fittest solution found by evolution 

        - 'best_fitness_per_generation' : np.ndarray
        A record of the best fitness value per generation (or in other words a learning curve), of shape (n_gen+1,)

        - 'mean_fitness_per_generation' : np.ndarray 
        A record of the mean fitness value per generation (of the pruned population if applied), of shape (n_gen+1,)

        - 'median_fitness_per_generation' : np.ndarray
        A record of the median fitness value per generation (of the pruned population if applied), of shape (n_gen+1,)

        - 'stdev_fitness_per_generation' : np.ndarray
        A record of the scatter of fitness values in terms of their standard deviation per generation (of the pruned population if applied), of shape (n_gen+1,)

        """

        if self._gen == 0:
            self.pop = pop
                
        self.pop_dec = pop_bin2dec(
            pop_bin = self.pop,
            n_bits_segment = self.n_bits_segment,
            decimal_acc = self.decimal_acc,
            offset = self.offset,
        )
        if n_workers > 1:
            with Pool(processes = n_workers) as pool:
                self.fitness_arr = np.array(pool.map(self.fitness_func, self.pop_dec))
        else:
            self.fitness_arr = self.fitness_func(self.pop_dec)

        if self._gen == 0:
            self.best_fitness_per_gen   = [np.max(self.fitness_arr)]
            self.mean_fitness_per_gen   = [np.mean(self.fitness_arr)]
            self.median_fitness_per_gen = [np.median(self.fitness_arr)]
            self.stdev_fitness_per_gen  = [np.std(self.fitness_arr)]

        if verbose:
            progress_bar = tqdm(total = n_gen, initial = self._gen, desc=f"[genetical] Evolution in progress", unit = "gen", file=sys.stdout,)

        indsort = np.argsort(self.fitness_arr)
        if elitist:
            self.elites = self.pop[indsort][-n_elites:]
            self.fitness_elites = self.fitness_arr[indsort][-n_elites:]

        if liberal:
            self.runts = self.pop[indsort][:n_runts]
            self.fitness_runts = self.fitness_arr[indsort][:n_runts]

        if self._gen < n_gen:
            for gen in range(self._gen, n_gen):
                if switch_selection is not None and gen >= switch_selection:
                    rank_selection = True
                elif switch_selection is not None and gen < switch_selection:
                    rank_selection = False
                elif switch_selection is None:
                    rank_selection = False
                mating_pool = self.create_mating_pool(
                    pop = self.pop,
                    fitness_arr = self.fitness_arr,
                    n_pairs = n_pairs,
                    selection_type = selection_type,
                    rank_selection = rank_selection,
                    sp_rank = sp_rank_selection,
                )
                offspring, self.fitness_arr = self.breed(
                    mating_pool = mating_pool,
                    crossover_type = crossover_type,
                    prob_mut = prob_mut,
                    prune = prune,
                    pruning_cutoff = pruning_cutoff,
                    return_fitness = True,
                    n_workers = n_workers,
                )
                if elitist: 
                    offspring   = np.concatenate([offspring, self.elites])
                    self.fitness_arr = np.concatenate([self.fitness_arr, self.fitness_elites])
                if liberal: 
                    offspring   = np.concatenate([offspring, self.runts])
                    self.fitness_arr = np.concatenate([self.fitness_arr, self.fitness_runts])
                
                self.best_fitness_per_gen.append(np.max(self.fitness_arr))
                self.mean_fitness_per_gen.append(np.mean(self.fitness_arr))
                self.median_fitness_per_gen.append(np.median(self.fitness_arr))
                self.stdev_fitness_per_gen.append(np.std(self.fitness_arr))
                
                indsort = np.argsort(self.fitness_arr)
                if elitist:
                    self.elites = offspring[indsort][-n_elites:]
                    self.fitness_elites = self.fitness_arr[indsort][-n_elites:]

                if liberal:
                    self.runts = offspring[indsort][:n_runts]
                    self.fitness_runts = self.fitness_arr[indsort][:n_runts]
                
                self.pop = offspring
                self._gen += 1
                if verbose:
                    progress_bar.update(1)
        else:
            warnings.warn(f"Your population has already been evolved for {self._gen} generations. If you want to evolve for longer, please enter `n_gen` > {self._gen}. Skipping this evolution call...")

        indsort = np.argsort(self.fitness_arr)
        fittest_indiv = self.pop[indsort][-1]
        fittest_indiv_dec = bin2dec(fittest_indiv, self.n_bits_segment, self.decimal_acc, offset = self.offset)
        return {
            'fittest_individual': fittest_indiv_dec,
            'best_overall_fitness': self.fitness_arr[indsort][-1],
            'best_fitness_per_generation': np.array(self.best_fitness_per_gen),
            'mean_fitness_per_generation': np.array(self.mean_fitness_per_gen),
            'median_fitness_per_generation': np.array(self.median_fitness_per_gen),
            'stdev_fitness_per_generation': np.array(self.stdev_fitness_per_gen),
        }