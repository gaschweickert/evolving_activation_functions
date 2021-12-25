import tensorflow.keras.backend as K
import tensorflow as tf
import random
from core_unit import CORE_UNIT

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

    '''
    Evaluates the candidate at given index through k-fold crossvalidation
    fitness_base = 0 (loss-based), 1 (accuracy_based)
    mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    '''
    def evaluate_candidate(self, candidate, k, train_epochs, model, mode, num_of_blocks, verbosity=0):
        average_val_results = model.k_fold_crossvalidation(candidate[0], k, train_epochs, mode, num_of_blocks, verbosity)
        candidate[1] = average_val_results[0] # average loss
        candidate[2] = average_val_results[1] # average accuracy
        return candidate
            
        
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
    N-sized population generator of complexity C, done on the basis on random generation.
    '''
    def generate_N_candidates(self):
        # creating population from candidate solutons
        self.population = []
        for i in range(self.N):
            self.population.append(self.generate_random_new_candidate_solution())
