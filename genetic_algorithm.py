import math
import itertools
import random

class GA:
    def __init__(self):
        self.unary_units = ['0', '1', 'x', '(-x)', 'abs(x)', 'x**2', 'x**3', 'sqrt(x)', 'exp(x)',  'exp(-x^2)',
        'log(1+exp(x))', 'log(abs(x + ?))', 'sin(x)', 'sinh(x)', 'asinh(x)',
        'cos(x)', 'cosh(x)', 'tanh(x)', 'atanh(x)', 'max(x, 0)', 'min(x, 0)',
        '(1/(1 + exp(-x)))', 'erf(x)', 'sinc(x)']
        self.binary_units = ['x1+x2', 'x1-x2', 'x1*x2', 'x1/(x2+?)', 'max(x1,x2)', 'min(x1,x2)']
        

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


    # N is the size of the population only depth 2 so far
    def population_initializer(self, N):
        population = []
        population_encoded = []
        ib = list(range(0, len(self.binary_units)))
        iu = list(range(0, len(self.unary_units)))
        a = list([ib, iu, iu])
        all_combos = list(itertools.product(*a))
        random.shuffle(all_combos)
        #print(all_combos)
        for i in range(N):
            binary_unit = self.binary_units[all_combos[i][0]]
            unary_unit1 = self.unary_units[all_combos[i][1]]
            unary_unit2 = self.unary_units[all_combos[i][2]]
            core_unit = self.core_unit_builder(binary_unit, unary_unit1, unary_unit2)
            population.append(core_unit)
            population_encoded.append([all_combos[i][0], all_combos[i][1], all_combos[i][2]])
        self.population = population
        self.population_encoded = population_encoded
