from genetic_algorithm_search import GAS
from random_search import RS
from cnn import CNN
import csv
import time
from search import SEARCH
from data import DATA
import datetime

def ga_search(dataset, generations, N, C, m, b, fitness_metric, train_epochs, mode, number_of_blocks, verbosity=0, save=True):
    date_and_time = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    t0 = time.time()

    cnn = CNN(dataset)
    gas = GAS(generations, N, C, m, b, fitness_metric)
    gas.run(train_epochs, cnn, mode, number_of_blocks, verbosity)

    t1 = time.time()
    total_time = t1-t0

    if save:
        base = '_loss-based_' if fitness_metric == 1 else '_accuracy-based_'
        save_file_name = date_and_time + '_GA-search' + base + dataset + '_G=' + str(generations) + '_N=' + str(N) + '_C=' + str(C) + '_m=' + str(m) + '_b=' + str(b) + '_mode=' + str(mode) + '_train-epochs=' + str(train_epochs) + '_number-of-blocks=' + str(number_of_blocks)
        gas.save_data_log(save_file_name, total_time)

    

def random_search(dataset, generations, N, C, train_epochs, mode, number_of_blocks, verbosity=0, save=True):
    date_and_time = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    t0 = time.time()

    cnn = CNN(dataset)
    rs = RS(generations, N, C)
    rs.run(train_epochs, cnn, mode, number_of_blocks, verbosity)

    t1 = time.time()
    total_time = t1-t0

    if save:
        save_file_name = date_and_time + '_Random-search'+ '_' + dataset + '_G=' + str(generations) + '_N=' + str(N) + '_C=' + str(C) + '_mode=' + str(mode) + '_train-epochs=' + str(train_epochs) + '_number-of-blocks=' + str(number_of_blocks)
        rs.save_data_log(save_file_name, total_time)

def test_candidate(dataset, candidate_keys, k, mode, no_blocks, no_epochs, verbosity=0, save_model=False, visualize=False, tensorboard_log=False):
    ss = SEARCH('None', 0,0,0)
    cnn= CNN(dataset)
    #candidate = ss.generate_candidate_solution_from_keys(candidate_keys)
    #candidate.loss, candidate.accuracy = cnn.final_test(k, mode, candidate.core_units, no_blocks, no_epochs, verbosity, save_model, visualize, tensorboard_log)
    loss, accuracy = cnn.final_test(k, mode, 'relu', no_blocks, no_epochs, verbosity, save_model, visualize, tensorboard_log)
    print(accuracy)
    #candidate.print_candidate_name_and_results()
    #candidate.plot_candidate()

#EARLY STOPPAGE AT EPOCH 148/200 (256 batch c1 gasearch jan 8) --> 1 gpu
#custom1: max(max(x, 0), log(abs(x + err)))
#Loss: 2.744981050491333; Accuracy: 0.7875999808311462
    


def main():
    # EXPERIMENT PARAMETERS
    # generations = 1
    # k = number of final test runs
    # N = population size (N-m-b>=2 for crossover)
    # C = search space complexity i.e. number of custom af (note: must match CNN arcchitecture)
    # m = number of new candidates per generation
    # b = number of preserved best candidates per generation
    # fitness_metric = 1 (loss) or 2 (accuracy) for fitness base metric

    # dataset = "cifar10" # 'cifar10' or 'cifar100'
    # mode = 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    # number_of_blocks = number of VGG blocks
    # number of layers = number of blocks * 2 + 1
    # train_epochs = number of training epochs


    ga_search(dataset = 'cifar100', generations=10, N=50, C=1, m=10, b=5, fitness_metric=1, train_epochs=50, mode=1, number_of_blocks=2, verbosity=0, save=True)
    #random_search(dataset = 'cifar10', generations=10, N=50, C=3, train_epochs=50, mode=3, number_of_blocks=2, verbosity=0, save=True)
    #test_candidate(dataset = 'cifar10', candidate_keys = [['max(x, 0)', 'max(x1, x2)', 'log(abs(x + err))']], k = 1, mode=1, no_blocks=2, no_epochs=200, verbosity=1, save_model=False, visualize=False, tensorboard_log=True)


    #data = DATA()
    #data.collect_data_from_file("search_data/08-Jan-2022_22:27:44_GA-search_loss-based_cifar10_G=10_N=50_C=1_m=10_b=5_mode=1_train-epochs=50_number-of-blocks=2.csv")
    #data.collect_data_from_file("search_data/09-Jan-2022_15:33:45_Random-search_cifar10_G=10_N=50_C=1_mode=1_train-epochs=50_number-of-blocks=2.csv")
    #data.collect_data_from_file("search_data/10-Jan-2022_00:49:53_GA-search_loss-based_cifar10_G=10_N=50_C=1_m=10_b=5_mode=1_train-epochs=50_number-of-blocks=2.csv")
    #data.plot_gen_vs_accuracy()
    #data.print_overall_best()
    #gasearch: 'max(x, 0)', 'max(x1, x2)', 'log(abs(x + err))'


    




if __name__ == "__main__":
    main()
