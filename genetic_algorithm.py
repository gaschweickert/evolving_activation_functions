import math
import itertools
import random
from core_unit import CORE_UNIT
import tensorflow.keras.backend as K
import tensorflow as tf

class GA:
    def __init__(self):
        self.population = []

        self.err = 0.000001

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
            'log(1+exp(x))' : lambda x:K.log(1+K.exp(x)), 
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
            'x1+x2' : lambda x1, x2:x1 + x2, 
            'x1-x2' : lambda x1, x2:x1 - x2, 
            'x1*x2' : lambda x1, x2:x1 * x2, 
            'x1/(x2+err)' : lambda x1, x2:x1/(x2 + self.err), 
            'max(x1,x2)' : lambda x1, x2:K.maximum(x1,x2), 
            'min(x1,x2)' : lambda x1, x2:K.minimum(x1,x2)

        }


    def print_population(self):
        print(self.population)

    def evaluate_candidate(self, candidate_idx, model):
        model.set_custom_activation(self.population[candidate_idx][0])
        model.build_and_compile()
        #model.summary()
        model.train_and_validate()
        results = model.evaluate()
        self.population[candidate_idx][1] = results[1] #uses accuracy for absolute fitness update
        print("test loss, test acc:", results)

    '''
    Softmax operation turning absolute fitness into sampling probabilities. Sampling 2(N-m)
    Selecting parents
    '''
    def evolve(self, N, m, d):
        # determining selection prob based on fitness
        fit_sum = 0
        for candidate in self.population:
            fit_sum+=candidate[1]
        
        selection_probabilities = []
        for candidate in self.population: 
            selection_probabilities.append(exp(candidate[1])/exp(fit_sum))

        # selecting parents 2*(N-m)
        parents = []
        for i in range(len(2*(N-m))):
            parents.append(random.choices(self.population, selection_probabilities))

        # crossover
        for i in 
            
            







    def crossover(self):


    '''
    params:
     N - size of the candidate solution population
     C - complextiy of candidate solutions (i.e. number of AFs)
    '''
    def population_initializer(self, N, C):
        unary_keys = list(self.unary_units)
        binary_keys = list(self.binary_units)
        a = list([binary_keys, unary_keys, unary_keys])
        all_combos = list(itertools.product(*a))
        random.shuffle(all_combos) 

        #print(all_combos)

        self.population = []
        candidate_solution = []
        fitness = 0
        for i in range(N*C):
            elementary_units_names = all_combos[i]
            elementary_units_functions = [self.binary_units[all_combos[i][0]], self.unary_units[all_combos[i][1]], self.unary_units[all_combos[i][2]]]
            core_unit = CORE_UNIT(elementary_units_names, elementary_units_functions)
            candidate_solution.append(core_unit)
            if ((i+1)%C == 0):
                self.population.append([candidate_solution, fitness])
                candidate_solution = []


    
