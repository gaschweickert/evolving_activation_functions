from genetic_algorithm_search import GAS
from random_search import RS
from cnn import CNN
import csv
import time
from search import SEARCH
import datetime

def ga_search(dataset, generations, N, C, m, b, fitness_metric, k, train_epochs, mode, number_of_blocks, save=True):
    date_and_time = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    t0 = time.time()

    cnn = CNN(dataset)
    gas = GAS(generations, N, C, m, b, fitness_metric)
    gas.run(k, train_epochs, cnn, mode, number_of_blocks)

    t1 = time.time()
    total_time = t1-t0

    if save:
        base = '_loss-based_' if fitness_metric == 1 else '_accuracy-based_'
        save_file_name = date_and_time + '_GA-search' + base + dataset + '_G=' + str(generations) + '_N=' + str(N) + '_C=' + str(C) + '_m=' + str(m) + '_b=' + str(b) + '_mode=' + str(mode) + '_k=' + str(k) + '_train-epochs=' + str(train_epochs) + '_number-of-blocks=' + str(number_of_blocks)
        gas.save_data_log(save_file_name, total_time)

    

def random_search(dataset, generations, N, C, k, train_epochs, mode, number_of_blocks, save=True):
    date_and_time = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    t0 = time.time()

    cnn = CNN(dataset)
    rs = RS(generations, N, C)
    rs.run(k, train_epochs, cnn, mode, number_of_blocks)

    t1 = time.time()
    total_time = t1-t0

    if save:
        save_file_name = date_and_time + '_Random-search'+ '_' + dataset + '_G=' + str(generations) + '_N=' + str(N) + '_C=' + str(C) + '_mode=' + str(mode) + '_k=' + str(k) + '_train-epochs=' + str(train_epochs) + '_number-of-blocks=' + str(number_of_blocks)
        rs.save_data_log(save_file_name, total_time)




def main():
    # EXPERIMENT PARAMETERS
    # generations = 1
    # k = number of folds for crossvalidation
    # N = population size (N-m-b>=2 for crossover)
    # C = search space complexity i.e. number of custom af (note: must change layer set up in CNN)
    # m = number of new candidates per generation
    # b = number of preserved best candidates per generation
    # fitness_metric = 1 (loss) or 2 (accuracy) for fitness base metric

    # dataset = "cifar10" # 'cifar10' or 'cifar100'
    # mode = 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    # number_of_blocks = number of layers = number of blocks * 2 (+ 1)
    # train_epochs = number of training epochs


    ga_search(dataset = 'cifar10', generations=2, N=2, C=2, m=0, b=0, fitness_metric=1, k=2, train_epochs=1, mode=3, number_of_blocks=2, save=True)
    #random_search(dataset = 'cifar10', generations=2, N=2, C=1, k=2, train_epochs=2, mode=1, number_of_blocks=1, save=True)









    '''
    #print("\nComparison:")
    #ss = SEARCH('None', generations, N, C)
    #test_epochs = 1

    #candidate = ss.generate_candidate_solution_from_keys([['abs(x)', 'x1 / (x2 + err)', '1']])
    candidate[1], candidate[2] = cnn.test(mode, candidate[0], number_of_blocks, test_epochs, verbose=1)
    ss.print_candidate_name_and_results(candidate)
    '''

    #benchmark = ['relu', 0.0, 0.0]
    #benchmark[1], benchmark[2] = cnn.test(mode, benchmark[0], number_of_blocks, test_epochs, verbose=1)
    #ss.print_candidate_name_and_results(benchmark)


    




if __name__ == "__main__":
    main()
