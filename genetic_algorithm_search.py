from search import SEARCH
import math
import random
import numpy as np
from core_unit import CORE_UNIT
import tensorflow as tf
import tensorflow.keras.backend as K
from operator import itemgetter

# seed random number generator
#seed(1)

class GAS(SEARCH):
    def __init__(self, generations, N, C, m, b, fitness_metric):
        super().__init__("GA SEARCH", generations, N, C)
        self.fitness_metric = fitness_metric
        self.m = m
        self.b = b

    def run(self, k, train_epochs, cnn, mode, number_of_blocks):
        self.generate_N_candidates() 
        gen_best_candidates = []
        for gen in range(1, self.generations + 1):
            print(self.search_type)
            for candidate_idx in range(self.N):
                print("\nGeneration #" + str(gen) + " : Candidate #" + str(candidate_idx + 1))
                self.print_candidate_name(self.get_candidate(candidate_idx))
                evaluated_candidate = self.evaluate_candidate(candidate_idx, k, train_epochs, cnn, mode, number_of_blocks, verbosity=1)
                self.print_candidate_results(evaluated_candidate)
            print("\nGeneration #" + str(gen) + ' : Best Candidate')
            gen_best_candidate = self.get_population_best_candidate(evaluation_metric=2) # best accuracy wise
            self.print_candidate_name_and_results(gen_best_candidate)
            gen_best_candidates.append(gen_best_candidate)
            #self.print_population()
            if (gen != self.generations): self.evolve(self.fitness_metric) # do not evolve final generation

        print('Every generation best:')
        for i, can in enumerate(gen_best_candidates):
            print("\nGeneration #" + str(i + 1))
            self.print_candidate_name_and_results(can)


    '''
    Softmax operation turning absolute fitness into sampling probabilities. Sampling 2(N-m)
    Selecting parents
    '''
    def evolve(self, fitness_metric):
        assert self.N - self.m - self.b >= 2, "Not enough parents for crossover!"
        assert fitness_metric == 1 or 2, "Invalid fitness metric should be 1 (loss) or 2 (accuracy)"

        new_population = []
        # selecting top b candidates to keep in population without mutation or crossover
        ordered_population = sorted(self.population, key=itemgetter(fitness_metric), reverse=False if fitness_metric == 1 else True)
        best_candidates = ordered_population[:self.b]
        for candidate in best_candidates:
            new_population.append(candidate)

        # determining selection prob based on fitness
        fitness_exp_sum  = 0
        for candidate in self.population:
            if not math.isnan(candidate[fitness_metric]):
                fitness_exp_sum += K.exp(candidate[fitness_metric])
        
        selection_probabilities = []
        valence = -1 if fitness_metric == 1 else 1 # higher values refer to fitter solutions with accuracy metric and the opposite is true for loss
        for candidate in self.population: 
            if not math.isnan(candidate[fitness_metric]):
                selection_probabilities.append(K.exp(valence * candidate[fitness_metric])/fitness_exp_sum)
            else:
                selection_probabilities.append(0.0) # if solution results in NaN loss give 0 change of reproducting

        # selecting parents 2*(N-m-b)
        num_of_parents = 2*(self.N-self.m-self.b)
        parents = random.choices(self.population, selection_probabilities, k=num_of_parents)

        # crossover and mutation
        for i in range(0, len(parents), 2):
            mutated_child = self.crossover_and_mutate(parents[i], parents[i+1])
            new_population.append(mutated_child)

        # add random candidate solutions for exploration
        for i in range(self.m):
            new_population.append(self.generate_random_new_candidate_solution())

        assert (len(self.population) == len(new_population))
        
        # reset evaluation metrics
        for candidate in new_population:
            candidate[1] = 0.0
            candidate[2] = 0.0

        self.population = new_population
        

    # Two methods of crossover
    def crossover_and_mutate(self, parent1, parent2):
        crossover_anywhere = True # if False crossover only happens between core units

        parent1_gene = parent1[0] # [core unit, core unit, ...]
        parent2_gene = parent2[0] 
        assert len(parent1_gene) == len(parent2_gene)

        length_of_core_unit = len(parent1_gene[0].get_elementary_units_keys()) #3
        length_of_gene = len(parent1_gene) * length_of_core_unit


        if crossover_anywhere:
            # getting all keys of elementary units of parent genes
            parent1_gene_keys = [core_unit.get_elementary_units_keys() for core_unit in parent1_gene] # [[unary_unit, binary_unit, unary_unit], ...]
            parent2_gene_keys = [core_unit.get_elementary_units_keys() for core_unit in parent2_gene]

            # detemining random gene crossover point
            gene_crossover_point = random.randint(0, length_of_gene - 1)

            # performing crossover (CUTS ANYWHERE)
            flat_parent1_gene_keys = np.array(parent1_gene_keys).flatten()
            flat_parent2_gene_keys = np.array(parent2_gene_keys).flatten()
            flat_child_gene_keys = np.concatenate((flat_parent1_gene_keys[:gene_crossover_point], flat_parent2_gene_keys[gene_crossover_point:]))
            child_gene_keys = flat_child_gene_keys.reshape((len(parent1_gene),length_of_core_unit)).tolist()

        else: # Only cross over between core_units
            assert self.C > 1, "Not enough core_units to perform cross over between core_units (C must be > 1)"
            gene_crossover_point = random.randint(0, len(parent1_gene) - 1)
            child_gene = parent1_gene[:gene_crossover_point] + parent2_gene[gene_crossover_point:]
            assert len(parent1_gene) == len(child_gene)
            child_gene_keys = [core_unit.get_elementary_units_keys() for core_unit in child_gene]

        # determining random gene mutation point
        gene_mutation_point = random.randint(0, length_of_gene - 1) 
        #print("gene_mutation_point = " + str(gene_mutation_point))

        # performing mutation
        core_unit_mutation_point = gene_mutation_point%length_of_core_unit
        if (core_unit_mutation_point == 1): # binary_unit
            mutated_unit_key = random.sample(list(self.binary_units), 1)[0]
        else:
            mutated_unit_key = random.sample(list(self.unary_units), 1)[0]
        child_gene_keys[int(gene_mutation_point//length_of_core_unit)][gene_mutation_point%length_of_core_unit] = mutated_unit_key

        # form core unit
        child_gene = []
        for core_unit_keys in child_gene_keys:
            unary1, binary, unary2 = core_unit_keys
            core_unit_functions = [self.unary_units[unary1], self.binary_units[binary], self.unary_units[unary2]]
            core_unit = CORE_UNIT(core_unit_keys, core_unit_functions)
            child_gene.append(core_unit)

        # reset fitness metrics
        loss = 0.0
        accuracy = 0.0
        
        return [child_gene, loss, accuracy]


    '''
    params:
     N - size of the candidate solution population
     C - complextiy of candidate solutions (i.e. number of AFs)
    # POPULATION INITIALIZATION IS DOES NOT USE RANDOM CANDDIATE GENERATOR
    def initialize(self, N, C, m):
        # set evolution parameters
        self.N = N
        self.C = C
        self.m = m

        # generating all possible af
        unary_keys = list(self.unary_units)
        binary_keys = list(self.binary_units)
        a = list([unary_keys, binary_keys, unary_keys])
        all_combos = list(itertools.product(*a))
        random.shuffle(all_combos) 


        # creating population from candidate solutons
        self.population = []
        candidate_solution = []
        loss = 0.0
        accuracy = 0.0
        for i in range(N*C):
            elementary_units_keys = all_combos[i]
            elementary_units_functions = [self.unary_units[all_combos[i][0]], self.binary_units[all_combos[i][1]], self.unary_units[all_combos[i][2]]]
            core_unit = CORE_UNIT(elementary_units_keys, elementary_units_functions)
            candidate_solution.append(core_unit)
            if ((i+1)%C == 0):
                self.population.append([candidate_solution, loss, accuracy])
                candidate_solution = []
        '''

    


    
