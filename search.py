from math import nan
import tensorflow.keras.backend as K
import tensorflow as tf
import random
import itertools
import math
import csv
import os
from core_unit import CORE_UNIT
from candidate import CANDIDATE

class SEARCH:

    def __init__(self, search_type, generations, N, C):
        self.search_type = search_type
        self.generations = generations
        self.N = N
        self.C = C

        self.population = []

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

        self.all_evaluated_candidate_solutions = []



    def get_candidate_at_idx(self, candidate_idx):
        return self.population[candidate_idx]

    def print_population(self):
        print("Entire Population:\n")
        for can in self.population:
            self.print_candidate_name_and_results(can)

    def get_population_best_candidate(self, evaluation_metric):
        if evaluation_metric == 1:
            valence = -1
            evaluation = lambda x: x.loss
        else:
            valence = 1
            evaluation = lambda x: x.accuracy
        best_candidate = self.population[0]
        for candidate in self.population:
            if valence * evaluation(candidate) > valence * evaluation(best_candidate):
                best_candidate = candidate
        return best_candidate


    '''
    Evaluates the candidate at given index through k-fold crossvalidation
    fitness_base = 0 (loss-based), 1 (accuracy_based)
    mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    '''
    def evaluate_candidate(self, candidate, train_epochs, model, mode, no_blocks, verbosity=0):
        if candidate.check_validity():        
            val_results = model.search_test(candidate.core_units, train_epochs, mode, no_blocks, verbosity)
        else:
            val_results = [nan, nan]
        candidate.loss = val_results[0] # loss
        candidate.accuracy = val_results[1] # accuracy

    # list of keys input should be in the following format: [[unary_key, binary_key, unary_key], ...]
    def generate_candidate_solution_from_keys(self, list_of_keys, loss=nan, accuracy=0.0):
        core_units = []
        for keys in list_of_keys:
            unary_unit1_key, binary_unit_key, unary_unit2_key = keys
            elementary_units_keys = keys
            elementary_units_functions = [self.unary_units[unary_unit1_key], self.binary_units[binary_unit_key], self.unary_units[unary_unit2_key]]
            core_unit = CORE_UNIT(elementary_units_keys, elementary_units_functions)
            #assert core_unit.check_validity(), "Invalid unit keys provided for candidate solution generation"
            core_units.append(core_unit)
        return CANDIDATE(core_units, loss, accuracy)

        
    '''
    Generates random candidate solution of complexity C
    '''
    def generate_random_new_candidate_solution(self):
        core_units = []
        i = self.C
        while (i):
            binary_unit_key = random.sample(list(self.binary_units), 1)[0]
            unary_unit1_key = random.sample(list(self.unary_units), 1)[0]
            unary_unit2_key = random.sample(list(self.unary_units), 1)[0]
            elementary_units_keys = [unary_unit1_key, binary_unit_key, unary_unit2_key]
            elementary_units_functions = [self.unary_units[unary_unit1_key], self.binary_units[binary_unit_key], self.unary_units[unary_unit2_key]]
            core_unit = CORE_UNIT(elementary_units_keys, elementary_units_functions)
            if core_unit.check_validity():
                core_units.append(core_unit)
                i = i - 1
        return CANDIDATE(core_units, loss=0.0, accuracy=0.0)

    def check_same_candidate_solution(self, can1, can2):
        same_cu = 0
        can1_cu_names = can1.get_candidate_name()
        can2_cu_names = can2.get_candidate_name()
        for i in range(len(can1_cu_names)):
            cu_i_can1_name = can1_cu_names[i]
            cu_i_can2_name = can2_cu_names[i]
            if cu_i_can1_name == cu_i_can2_name:
                same_cu = same_cu + 1
        return True if same_cu == len(can1_cu_names) else False
            

    def generate_n_unique_candidates(self, n):
        # creating population from unique random candidate solutons
        candidate_list = []
        while n:
            new = 1
            new_candidate = self.generate_random_new_candidate_solution()
            for existing_candidate in candidate_list:
                if self.check_same_candidate_solution(new_candidate, existing_candidate):
                    new = 0
                    n = n + 1
                    break
            if new:
                candidate_list.append(new_candidate)
                n = n-1
        return candidate_list
        
    def get_search_top_candidates(self, no_candidates, evaluation_metric):
        assert evaluation_metric in (1,2), 'Invalid evaluation metric'
        if evaluation_metric == 1:
            fitness = lambda x: float('inf') if math.isnan(x.loss) else x.loss
        else:
            fitness = lambda x: float('-inf') if math.isnan(x.accuracy) else x.accuracy

        '''
        fitness = lambda x: x.loss if evaluation_metric == 1 else lambda x: x.accuracy
        
        # selecting top no candidates to keep in population without mutation or crossover
        # arr[np.argsort(arr[:,1])]
        nan_removed_population = []
        nan_population = []
        for candidate in self.population:
            if not math.isnan(fitness(candidate)):
                nan_removed_population.append(candidate)
            else:
                nan_population.append(candidate)
        nan_removed_ordered_population = sorted(nan_removed_population, key=fitness, reverse=False if evaluation_metric == 1 else True)
        ordered_population = nan_removed_ordered_population + nan_population
        return ordered_population[:no_candidates]
        '''
        ordered_population = sorted(self.population, key=fitness, reverse=False if evaluation_metric == 1 else True)
        return ordered_population[:no_candidates]

        



    def save_data_log(self, save_file_name, time_taken=0):
        assert self.all_evaluated_candidate_solutions, "Evaluated candidate solutions list is empty"
        fields = ['Gen']
        for i in range(1, self.C + 1):
            fields.append('C'+ str(i) + '_name')
            fields.append('C'+ str(i) + '_unary1_key')
            fields.append('C'+ str(i) + '_binary_key')
            fields.append('C'+ str(i) + '_unary2_key')
        fields.extend(['Loss', 'Accuracy'])

        filepath = os.path.join('./', 'search_data', save_file_name + '.csv')
        with open(filepath, 'w') as f:
            
            # using csv.writer method from CSV package
            write = csv.writer(f)
            
            write.writerow(fields)

            gen = 1
            for i, candidate in enumerate(self.all_evaluated_candidate_solutions):
                entry = [gen]
                for cu in candidate.core_units:
                    entry.append(cu.get_name())
                    entry.extend(cu.get_elementary_units_keys())
                entry.extend([candidate.loss, candidate.accuracy])
                write.writerow(entry)
                if not (i + 1) % self.N: gen = gen + 1

            write.writerow(['Total time:',time_taken])



 
