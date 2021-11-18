import math
import itertools
import random

class GA:
    def __init__(self):
        self.unary_units = ['0', '1', 'x', '(-x)', 'abs(x)', 'x**2', 'x**3', 'sqrt(x)', 'exp(x)',  'exp(-x^2)',
        'log(1+exp(x))', 'log(abs(x + err))', 'sin(x)', 'sinh(x)', 'asinh(x)',
        'cos(x)', 'cosh(x)', 'tanh(x)', 'atanh(x)', 'max(x, 0)', 'min(x, 0)',
        '(1/(1 + exp(-x)))', 'erf(x)', 'sinc(x)']
        self.binary_units = ['x1+x2', 'x1-x2', 'x1*x2', 'x1/(x2+err)', 'max(x1,x2)', 'min(x1,x2)']
        

        self.population = []
        self.population_encoded = []

    def core_unit_builder(self, binary_u, unary_u1, unary_u2):
        binary_u = binary_u.replace("x1", unary_u1)
        binary_u = binary_u.replace("x2", unary_u2)
        return binary_u


    def get_population(self):
        return self.population
    
    def get_encoded_population(self):
        return self.population_encoded

    '''

    params:
     N - size of the candidate solution population
     C - complextiy of candidate solutions (i.e. number of AFs)
    '''
    def population_initializer(self, N, C):
        ib = list(range(0, len(self.binary_units)))
        iu = list(range(0, len(self.unary_units)))
        a = list([ib, iu, iu])
        all_combos = list(itertools.product(*a))
        random.shuffle(all_combos) #list of inidices
        print(all_combos)

        self.population = []
        self.population_encoded = []
        candidate_solution = []
        candidate_solution_encoded = []
        for i in range(N*C):
                binary_unit = self.binary_units[all_combos[i][0]]
                unary_unit1 = self.unary_units[all_combos[i][1]]
                unary_unit2 = self.unary_units[all_combos[i][2]]
                core_unit = self.core_unit_builder(binary_unit, unary_unit1, unary_unit2)
                candidate_solution.append(core_unit) #text
                candidate_solution_encoded.append([all_combos[i][0], all_combos[i][1], all_combos[i][2]])
                if ((i+1)%C == 0):
                    self.population.append(candidate_solution)
                    self.population_encoded.append(candidate_solution_encoded)
                    candidate_solution = []
                    candidate_solution_encoded = []


    
