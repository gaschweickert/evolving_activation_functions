import math
import itertools
import random
from core_unit import CORE_UNIT
import tensorflow.keras.backend as K
import tensorflow as tf

# seed random number generator
#seed(1)

class GA:
    def __init__(self):
        self.population = []
        self.N = None
        self.C = None
        self.m = None

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


    def print_population(self):
        print(self.population)

    def print_candidate_solution(self, sol):
        for i in range(len(sol[0])):
            print(sol[0][i].get_name())
        print('Accuracy:' + str(sol[1]))

    def get_population_best_candidate(self):
        best_candidate = ['NONE', 0]
        for candidate in self.population:
            if candidate[1] > best_candidate[1]:
                best_candidate = candidate
        return best_candidate

    def evaluate_candidate(self, candidate_idx, model, custom):
        if custom: model.set_custom_activation(self.population[candidate_idx][0])
        model.build_and_compile(custom)
        #model.summary()
        model.train_and_validate()
        results = model.evaluate()
        if custom: self.population[candidate_idx][1] = results[1] #uses accuracy for absolute fitness update
        return self.population[candidate_idx] if custom else ['Relu', results[1]]

    '''
    Softmax operation turning absolute fitness into sampling probabilities. Sampling 2(N-m)
    Selecting parents
    '''
    def evolve(self):
        assert self.N >= 2

        # determining selection prob based on fitness
        fit_sum = 0
        for candidate in self.population:
            fit_sum+=candidate[1]
        
        selection_probabilities = []
        for candidate in self.population: 
            selection_probabilities.append(K.exp(candidate[1])/K.exp(fit_sum))

        # selecting parents 2*(N-m)
        parents = []
        for i in range(2*(self.N-self.m)):
            parents.append(random.choices(self.population, selection_probabilities)[0])

        new_population = []
        # crossover and mutation
        for i in range(0, len(parents), 2):
            mutated_child = self.crossover_and_mutate(parents[i], parents[i+1])
            new_population.append(mutated_child)

        # add random candidate solutions for exploration
        for i in range(self.m):
            new_population.append(self.generate_random_new_candidate_solution())

        assert (len(self.population) == len(new_population))

        self.population = new_population
        

    # CROSSOVER IS WEIRD!!
    def crossover_and_mutate(self, parent1, parent2):
        parent1_gene = parent1[0] #[core unit, core unit]
        parent2_gene = parent2[0] 
        assert len(parent1_gene) == len(parent2_gene)
        parent1_elem_units_keys = []
        parent2_elem_units_keys = []
        # getting all keys of elementary units of parent genes
        for i in range(len(parent1_gene)):
            parent1_elem_units_keys.append(parent1_gene[i].get_elementary_units_keys())
            parent2_elem_units_keys.append(parent2_gene[i].get_elementary_units_keys())
        
        # detemining random gene crossover point
        length_of_gene = len(parent1_gene) * len(parent1_gene[0].get_elementary_units_keys())
        gene_crossover_point = random.randint(0, length_of_gene - 1)
        #print("gene_crossover_point = " + str(gene_crossover_point))

        # performing crossover (WEIRD --> CUTS ANYWHERE)
        child_elem_units_keys = parent1_elem_units_keys
        length_of_core_unit = len(parent1_elem_units_keys[0]) #3
        for i in range(len(parent1_elem_units_keys)):
            for j in range(length_of_core_unit):
                if (j + i * length_of_core_unit >= gene_crossover_point):
                    child_elem_units_keys[i][j] = parent2_elem_units_keys[i][j]

        # determining random gene mutation point
        gene_mutation_point = random.randint(0, length_of_gene - 1) 
        #print("gene_mutation_point = " + str(gene_mutation_point))

        # performing mutation
        core_unit_mutation_point = gene_mutation_point%length_of_core_unit
        if (core_unit_mutation_point == 0): # binary_unit
            mutated_unit_key = random.sample(list(self.binary_units), 1)[0]
        else:
            mutated_unit_key = random.sample(list(self.unary_units), 1)[0]
        child_elem_units_keys[int(gene_mutation_point/length_of_core_unit)][gene_mutation_point%length_of_core_unit] = mutated_unit_key
    
        # form core unit
        child_gene = []
        for core_unit_keys in child_elem_units_keys:
            binary, unary1, unary2 = core_unit_keys
            core_unit_functions = [self.binary_units[binary], self.unary_units[unary1], self.unary_units[unary2]]
            core_unit = CORE_UNIT(core_unit_keys, core_unit_functions)
            child_gene.append(core_unit)

        # set fitness to 0
        fitness = 0
        
        return [child_gene, fitness]

    '''
    Generates random candidate solution of complexity C
    '''
    def generate_random_new_candidate_solution(self):
        candidate_solution = []
        for af in range(self.C):
            binary_unit_key = random.sample(list(self.binary_units), 1)[0]
            unary_unit1_key = random.sample(list(self.unary_units), 1)[0]
            unary_unit2_key = random.sample(list(self.unary_units), 1)[0]
            elementary_units_keys = [binary_unit_key, unary_unit1_key, unary_unit2_key]
            elementary_units_functions = [self.binary_units[binary_unit_key], self.unary_units[unary_unit1_key], self.unary_units[unary_unit2_key]]
            core_unit = CORE_UNIT(elementary_units_keys, elementary_units_functions)
            candidate_solution.append(core_unit)
        fitness = 0
        return [candidate_solution, fitness]


    '''
    params:
     N - size of the candidate solution population
     C - complextiy of candidate solutions (i.e. number of AFs)
    '''
    def initialize(self, N, C, m):
        # set evolution parameters
        self.N = N
        self.C = C
        self.m = m

        # generating all possible af
        unary_keys = list(self.unary_units)
        binary_keys = list(self.binary_units)
        a = list([binary_keys, unary_keys, unary_keys])
        all_combos = list(itertools.product(*a))
        random.shuffle(all_combos) 


        # creating population from candidate solutons
        self.population = []
        candidate_solution = []
        fitness = 0
        for i in range(N*C):
            elementary_units_keys = all_combos[i]
            elementary_units_functions = [self.binary_units[all_combos[i][0]], self.unary_units[all_combos[i][1]], self.unary_units[all_combos[i][2]]]
            core_unit = CORE_UNIT(elementary_units_keys, elementary_units_functions)
            candidate_solution.append(core_unit)
            if ((i+1)%C == 0):
                self.population.append([candidate_solution, fitness])
                candidate_solution = []


    
