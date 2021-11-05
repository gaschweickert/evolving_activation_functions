import math
import itertools
import random
import numpy as np

class GA:
    def __init__(self):
        self.unary_units = ['0', '1', 'x', '(-x)', 'abs(x)', 'x**2', 'x**3', 'sqrt(x)', 'math.exp(x)',  'math.exp(-x^2)',
        'math.log(1+math.exp(x))', 'math.log(abs(x + ?))', 'math.sin(x)', 'math.sinh(x)', 'math.arcsinh(x)',
        'math.cos(x)', 'math.cosh(x)', 'math.tanh(x)', 'math.archtanh(x)', 'math.max(x, 0)', 'math.min(x, 0)',
        '(1/(1 + math.exp(-x)))', 'math.erf(x)', 'sinc(x)']
        self.binary_units = ['x1+x2', 'x1-x2', 'x1*x2', 'x1/(x2+?)', 'math.max(x1,x2)', 'min{x1,x2}']
        self.population = []

    def string_to_func(string, x):
        string.replace("x", str(x))
        print(string)
        return eval(string)

    def core_unit_builder(self, binary_u, unary_u1, unary_u2):
        binary_u = binary_u.replace("x1", unary_u1)
        binary_u = binary_u.replace("x2", unary_u2)
        return binary_u

    def get_population(self):
        return self.population


    #def core_unit(ib, iu1, iu2):
        #core_unit = core_unit_builder(binary_units[ib], unary_units[iu1], unary_units[iu2])
        #print(string_to_func(core_unit, x))

    # N is the size of the population
    # d is the depth of the tree
    def population_initializer(self, N, d):
        population = []
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
        self.population = population
