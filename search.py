import tensorflow.keras.backend as K
import tensorflow as tf
import random
import itertools
from core_unit import CORE_UNIT
import csv
from operator import itemgetter
import os

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
            '-x)' : lambda x:(-x), 
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
        if  isinstance(sol[0], str): 
            print(sol[0])
        else:
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


    '''
    Evaluates the candidate at given index through k-fold crossvalidation
    fitness_base = 0 (loss-based), 1 (accuracy_based)
    mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    '''
    def evaluate_candidate(self, candidate, train_epochs, model, mode, no_blocks, verbosity=0):
        average_val_results = model.search_test(candidate[0], train_epochs, mode, no_blocks, verbosity)
        candidate[1] = average_val_results[0] # average loss
        candidate[2] = average_val_results[1] # average accuracy

    # list of keys input should be in the following format: [[unary_key, binary_key, unary_key], ...]
    def generate_candidate_solution_from_keys(self, list_of_keys):
        candidate_solution = []
        for keys in list_of_keys:
            unary_unit1_key, binary_unit_key, unary_unit2_key = keys
            elementary_units_keys = keys
            elementary_units_functions = [self.unary_units[unary_unit1_key], self.binary_units[binary_unit_key], self.unary_units[unary_unit2_key]]
            core_unit = CORE_UNIT(elementary_units_keys, elementary_units_functions)
            assert core_unit.check_validity(), "Invalid unit keys provided for candidate solution generation"
            candidate_solution.append(core_unit)
        accuracy = 0.0
        loss = 0.0
        return [candidate_solution, loss, accuracy]

        
    '''
    Generates random candidate solution of complexity C
    '''
    def generate_random_new_candidate_solution(self):
        candidate_solution = []
        i = self.C
        while (i):
            binary_unit_key = random.sample(list(self.binary_units), 1)[0]
            unary_unit1_key = random.sample(list(self.unary_units), 1)[0]
            unary_unit2_key = random.sample(list(self.unary_units), 1)[0]
            elementary_units_keys = [unary_unit1_key, binary_unit_key, unary_unit2_key]
            elementary_units_functions = [self.unary_units[unary_unit1_key], self.binary_units[binary_unit_key], self.unary_units[unary_unit2_key]]
            core_unit = CORE_UNIT(elementary_units_keys, elementary_units_functions)
            if core_unit.check_validity():
                candidate_solution.append(core_unit)
                i = i - 1
        accuracy = 0.0
        loss = 0.0
        return [candidate_solution, loss, accuracy]

    def same_candidate_solution(self, can1, can2):
        same_cu = 0
        for i in range(self.C):
            cu_i_can1_name = can1[0][i].get_name()
            cu_i_can2_name = can2[0][i].get_name()
            if cu_i_can1_name == cu_i_can2_name:
                same_cu = same_cu + 1
        return True if same_cu == self.C else False
            

    def generate_n_unique_candidates(self, n):
        # creating population from unique random candidate solutons
        candidate_list = []
        while n:
            new = 1
            new_candidate = self.generate_random_new_candidate_solution()
            for existing_candidate in candidate_list:
                if self.same_candidate_solution(new_candidate, existing_candidate):
                    new = 0
                    n = n + 1
                    break
            if new:
                candidate_list.append(new_candidate)
                n = n-1
        return candidate_list
        



    def get_search_top_candidates(self, number_of_candidates=3, evaluation_metric = 2):
        ordered_search_candidates = sorted(self.all_evaluated_candidate_solutions, key=itemgetter(evaluation_metric), reverse=False if evaluation_metric == 1 else True)
        return ordered_search_candidates[:number_of_candidates]

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
                for af in candidate[0]:
                    entry.append(af.get_name())
                    entry.extend(af.get_elementary_units_keys())
                entry.extend([candidate[1], candidate[2]])
                write.writerow(entry)
                if (i)%self.N: gen = gen + 1

            write.writerow(['Total time:',time_taken])



 
