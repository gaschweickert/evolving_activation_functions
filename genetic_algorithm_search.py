import math
import itertools
import random
import numpy as np
from core_unit import CORE_UNIT
import tensorflow as tf
import tensorflow.keras.backend as K
from operator import itemgetter

# seed random number generator
#seed(1)

class GA:
    def __init__(self, N, C, m, b):
        self.population = []
        self.N = N
        self.C = C
        self.m = m
        self.b = b

        self.err = 0.0000000000000000000000000001

        self.unary_units = {
            '0': 0.0, 
            '1': 1.0, 
            'x' : lambda x:x, 
            '(-x)' : lambda x:(-x), 
            'abs(x)' : lambda x:K.abs(x),
            'x**2' : lambda x:x**2, 
            'x**3' : lambda x:x**3,
            'sqrt(x)' : lambda x:K.sqrt(x), 
            'exp(x)' : lambda x:K.exp(x),  
            'exp(-x^2)' : lambda x:K.exp(-x**2),
            'log(1 + exp(x))' : lambda x:K.log(1+K.exp(x)), 
            'log(abs(x + err))' : lambda x:K.log(K.abs(x + self.err)), 
            'sin(x)' : lambda x:K.sin(x), 
            'sinh(x)' : lambda x:tf.math.sinh(x), 
            'asinh(x)' : lambda x:tf.math.asinh(x),
            'cos(x)' : lambda x:K.cos(x),
            'cosh(x)' : lambda x:tf.math.cosh(x), 
            'tanh(x)' : lambda x:tf.math.tanh(x), 
            'atanh(x)' : lambda x:tf.math.atanh(x), 
            'max(x, 0)' : lambda x:K.maximum(x, 0.0), 
            'min(x, 0)' : lambda x:K.minimum(x, 0.0),
            '(1/(1 + exp(-x)))' : lambda x:(1/(1 + K.exp(-x))), 
            'erf(x)' : lambda x:tf.math.erf(x), 
            'sinc(x)' : lambda x:K.sin(x)/(x + self.err) #sinc
        }

        self.binary_units = {
            'x1 + x2' : lambda x1, x2:x1 + x2, 
            'x1 - x2' : lambda x1, x2:x1 - x2, 
            'x1 * x2' : lambda x1, x2:x1 * x2, 
            'x1 / (x2 + err)' : lambda x1, x2:x1/(x2 + self.err), 
            'max(x1, x2)' : lambda x1, x2:K.maximum(x1,x2), 
            'min(x1, x2)' : lambda x1, x2:K.minimum(x1,x2)

        }

    def get_candidate(self, candidate_idx):
        return self.population[candidate_idx]

    def print_population(self):
        print("Entire Population:\n")
        for can in self.population:
            self.print_candidate_name_and_results(can)

    def print_candidate_name_and_results(self, sol):
        self.print_candidate_name(sol)
        self.print_candidate_results(sol)


    def print_candidate_name(self, sol):
        for i in range(len(sol[0])):
            print(sol[0][i].get_name())

    def print_candidate_results(self, sol):
        print('Loss: ' + str(sol[1]) + '; Accuracy: ' + str(sol[2]) + '\n')

    def get_population_best_candidate(self, evaluation_metric):
        valence = -1 if evaluation_metric == 1 else 1 
        best_candidate = self.population[0]
        for candidate in self.population:
            if candidate[valence * evaluation_metric] > best_candidate[valence * evaluation_metric]:
                best_candidate = candidate
        return best_candidate

    # fitness_base = 0 (loss-based), 1 (accuracy_based)
    # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    def evaluate_candidate(self, candidate_idx, k, train_epochs, model, mode, num_of_blocks, verbosity=0):
        if mode: model.set_custom_activation(self.population[candidate_idx][0])
        average_val_results = model.k_fold_crossvalidation_evaluation(k, train_epochs, model, mode, num_of_blocks, verbosity)
        if mode: # custom
            self.population[candidate_idx][1] = average_val_results[0] # average loss
            self.population[candidate_idx][2] = average_val_results[1] # average accuracy
            return self.population[candidate_idx]
        else:
            return ['Relu', average_val_results[0], average_val_results[1]]

    #def final_test(self):


    '''
    Softmax operation turning absolute fitness into sampling probabilities. Sampling 2(N-m)
    Selecting parents
    '''
    def evolve(self, evaluation_metric):
        assert self.N - self.m - self.b >= 2, "Not enough parents for crossover!"
        assert evaluation_metric == 1 or 2, "Invalid fitness metric should be 1 (loss) or 2 (accuracy)"

        new_population = []
        # selecting top b candidates to keep in population without mutation or crossover
        ordered_population = sorted(self.population, key=itemgetter(evaluation_metric), reverse=False if evaluation_metric == 1 else True)
        best_candidates = ordered_population[:self.b]
        for candidate in best_candidates:
            new_population.append(candidate)

        # determining selection prob based on fitness
        fitness_exp_sum  = 0
        for candidate in self.population:
            if not math.isnan(candidate[evaluation_metric]):
                fitness_exp_sum += K.exp(candidate[evaluation_metric])
        
        selection_probabilities = []
        valence = -1 if evaluation_metric == 1 else 1 # higher values refer to fitter solutions with accuracy metric and the opposite is true for loss
        for candidate in self.population: 
            if not math.isnan(candidate[evaluation_metric]):
                selection_probabilities.append(K.exp(valence * candidate[evaluation_metric])/fitness_exp_sum)
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
    Generates random candidate solution of complexity C
    '''
    def generate_random_new_candidate_solution(self):
        candidate_solution = []
        i = 0
        while (self.C - i):
            binary_unit_key = random.sample(list(self.binary_units), 1)[0]
            unary_unit1_key = random.sample(list(self.unary_units), 1)[0]
            unary_unit2_key = random.sample(list(self.unary_units), 1)[0]
            elementary_units_keys = [unary_unit1_key, binary_unit_key, unary_unit2_key]
            elementary_units_functions = [self.unary_units[unary_unit1_key], self.binary_units[binary_unit_key], self.unary_units[unary_unit2_key]]
            core_unit = CORE_UNIT(elementary_units_keys, elementary_units_functions)
            if core_unit.check_validity():
                candidate_solution.append(core_unit)
                i = i + 1
        accuracy = 0.0
        loss = 0.0
        return [candidate_solution, loss, accuracy]


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

    '''
    Parameter and initial population inititializer, done on the basis on random generation.

    params:
     N - size of the candidate solution population
     C - complextiy of candidate solutions (i.e. number of AFs)
     m - number of candidate solutions added each generation
    '''
    def initialize(self):
        # creating population from candidate solutons
        self.population = []
        for i in range(self.N):
            self.population.append(self.generate_random_new_candidate_solution())


    
