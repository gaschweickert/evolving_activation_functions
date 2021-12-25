from genetic_algorithm_search import GAS
from random_search import RS
from cnn import CNN
import csv
import time
from search import SEARCH

'''
def experiment1():
    generations = 10
    k = 5 # number of folds for crossvalidation
    N = 50 # population size (N-m-b>=2 for crossover)
    C = 1 # search space complexity i.e. number of custom af (note: must change layer set up in CNN)
    m = 10 # number of new candidates per generation
    b = 5 # number of preserved best candidates per generation
    fitness_metric = 1 # 1 (loss) or 2 (accuracy) for fitness base metric

    dataset = "cifar10" # 'cifar10' or 'cifar100'
    mode = 1 # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    number_of_blocks = 1 # number of layers = number of blocks * 2 (+ 1)
    train_epochs = 1 #number of training epochs

    cnn = CNN(dataset)

    rs = RS(generations, N, C)
    rs.run(k, train_epochs, cnn, mode, number_of_blocks)

    gas = GAS(generations, N, C, m, b, fitness_metric)
    gas.run(k, train_epochs, cnn, mode, number_of_blocks)
'''

def main():
    t0 = time.time()

    save_file = False
    
    generations = 10
    k = 2 # number of folds for crossvalidation
    N = 2 # population size (N-m-b>=2 for crossover)
    C = 2 # search space complexity i.e. number of custom af (note: must change layer set up in CNN)
    m = 0 # number of new candidates per generation
    b = 0 # number of preserved best candidates per generation
    fitness_metric = 1 # 1 (loss) or 2 (accuracy) for fitness base metric

    dataset = "cifar10" # 'cifar10' or 'cifar100'
    mode = 2 # 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    number_of_blocks = 1 # number of layers = number of blocks * 2 (+ 1)
    train_epochs = 1 #number of training epochs

    cnn = CNN(dataset)

    print("\nComparison:")
    ss = SEARCH('None', generations, N, C)
    test_epochs = 1

    #candidate = ss.generate_candidate_solution_from_keys([['abs(x)', 'x1 / (x2 + err)', '1']])
    candidate = ss.generate_random_new_candidate_solution()
    candidate[1], candidate[2] = cnn.test(mode, candidate[0], number_of_blocks, test_epochs, verbose=1)
    ss.print_candidate_name_and_results(candidate)

    #benchmark = ['relu', 0.0, 0.0]
    #benchmark[1], benchmark[2] = cnn.test(mode, benchmark[0], number_of_blocks, test_epochs, verbose=1)
    #ss.print_candidate_name_and_results(benchmark)



    #rs = RS(generations, N, C)
    #rs.run(k, train_epochs, cnn, mode, number_of_blocks)

    #gas = GAS(generations, N, C, m, b, fitness_metric)
    #gas.run(k, train_epochs, cnn, mode, number_of_blocks)
    
    t1 = time.time()
    total_time = t1-t0
    print("Time taken: " + str(total_time))


    #print("\nFinal best generated solution: (highest accuracy from all generations)")
    #final_best_candidate = max(ga.get_population_best_candidate(evaluation_metric))
    #ga.print_candidate_name_and_results(final_best_candidate)

    '''
    # field names 
    fields = ['Gen', 'Candidate', 'Loss', 'Acc']

    if save_file:
        with open('CNN_MNIST'+ '_' + dataset + '_N=' + str(N) + '_C=' + str(C) + '_G=' + str(generations) + '_m=' + str(m) + '_b=' + str(b), 'w') as f:
            
            # using csv.writer method from CSV package
            write = csv.writer(f)
            
            write.writerow(fields)
            for gen, candidate in enumerate(gen_best_candidates(evaluation_metric=2)): 
                candidate_name = str(';'.join([cu.get_name() for cu in candidate[0]]))
                write.writerow([gen, candidate_name, candidate[1]])
            write.writerow([999, relu_benchmark[0], relu_benchmark[1]])
    '''





if __name__ == "__main__":
    main()
