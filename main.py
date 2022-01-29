import os
from genetic_algorithm_search import GAS
from random_search import RS
from cnn import CNN
import csv
import time
from search import SEARCH
from data import DATA
import datetime
import numpy as np

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

# must all be from same original file
def test_candidates(filename, candidate_list, dataset, k, mode, no_blocks, no_epochs, big_nn_transfer=False, verbose=0, save_model=False, visualize=False, tensorboard_log=False, save_results=False):
    assert mode in (1,3), "Test_candidates not compatible with mode 2"
    cnn= CNN(dataset)
    candidate_results = []
    C = candidate_list[0].get_candidate_complexity()
    if verbose: print(filename)
    for candidate in candidate_list:
        candidate.print_candidate_name()
        if not no_blocks == C - 1 and mode == 3: #incompatible with layer mode (2)
            print("SOLUTION TRANSFERED TO")
            candidate.enlarge(no_blocks)
            candidate.print_candidate_name()
        candidate_results.append(cnn.final_test(k, mode, candidate.core_units, no_blocks, no_epochs, verbose, save_model, visualize, tensorboard_log))

    if save_results:
        for i, candidate in enumerate(candidate_list):
            save_file_name = dataset + "_no-block_" + str(no_blocks) + "_final_test_top" + str(i + 1) + "_k=" + str(k) + '_' + filename[12:-4]

            fields = ["k", "run_max_val_acc_index", "run_max_val_acc", "run_final_val_acc"]

            filepath = os.path.join('./', 'test_data', save_file_name + '.csv')
            with open(filepath, 'w') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)   
                write.writerow(fields)
                for r in candidate_results[i]:
                    write.writerow(r)
                write.writerow(candidate.get_candidate_name())

def test_benchmarks(dataset, k, no_blocks, no_epochs, verbosity, save_model=False, visualize=False, tensorboard_log=False, save_results=False):
    cnn = CNN(dataset)
    benchmarks = ["relu", "swish"] # swish
    benchmarks_results = []
    for benchmark_activation in benchmarks:
        print(benchmark_activation)
        benchmarks_results.append(cnn.final_test(k, 1, benchmark_activation, no_blocks, no_epochs, verbosity, save_model, visualize, tensorboard_log))
    
    if save_results:
        for i, k_results in enumerate(benchmarks_results):
            save_file_name = "final_test_" + str(benchmarks[i]) + "_" + dataset + "_k=" + str(k)+ "_no-blocks=" + str(no_blocks) + "_no_epochs=" + str(no_epochs)

            fields = ["k", "run_max_val_acc_index", "run_max_val_acc", "run_final_val_acc"]

            filepath = os.path.join('./', 'benchmark_data', save_file_name + '.csv')
            with open(filepath, 'w') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)
                write.writerow(fields)
                for r in k_results:
                    write.writerow(r)


def load_data(data):
    data.collect_data_from_file("search_data/15-Jan-2022_12:12:09_GA-search_loss-based_cifar10_G=15_N=50_C=3_m=10_b=5_mode=3_train-epochs=50_number-of-blocks=2.csv")
    data.collect_data_from_file("search_data/16-Jan-2022_14:05:33_GA-search_loss-based_cifar10_G=15_N=50_C=1_m=10_b=5_mode=1_train-epochs=50_number-of-blocks=2.csv")
    data.collect_data_from_file("search_data/18-Jan-2022_10:53:37_Random-search_cifar10_G=15_N=50_C=3_mode=3_train-epochs=50_number-of-blocks=2.csv")
    data.collect_data_from_file("search_data/18-Jan-2022_19:03:17_Random-search_cifar10_G=15_N=50_C=1_mode=1_train-epochs=50_number-of-blocks=2.csv")


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

    #ga_search(dataset = 'cifar10', generations=15, N=50, C=3, m=10, b=5, fitness_metric=1, train_epochs=50, mode=3, number_of_blocks=2, verbosity=0, save=True)
    
    #random_search(dataset = 'cifar10', generations=15, N=50, C=1, train_epochs=50, mode=1, number_of_blocks=2, verbosity=0, save=True)
    #test_candidate(dataset = 'cifar10', candidate_keys = [['max(x, 0)', 'max(x1, x2)', 'log(abs(x + err))']], k = 1, mode=1, no_blocks=2, no_epochs=200, verbosity=1, save_model=False, visualize=False, tensorboard_log=True)
    
    data = DATA()
    load_data(data)
    data.convert_and_order()
    data_n_tops = data.get_n_top_candidates(3, verbose=0)
    """
    for can_list in data_n_tops:
        for can in can_list:
            for cu in can.core_units:
                print(cu.get_elementary_units_keys())
            print()
    """


    #data.plot_gen_vs_accuracy()
    data_n_tops = data.get_n_top_candidates(3, verbose=0)
    for i, exp_n_tops in enumerate(data_n_tops):
        filename = data.filenames[i]
        split_name = filename.split("_")
        no_blocks = int(split_name[-1][-5])
        mode = int(split_name[-3][-1])
        #test_candidates(filename=filename, candidate_list=exp_n_tops, dataset='cifar100', k=5, mode=mode, no_blocks=2, no_epochs=200, verbose=0, save_model=False, visualize=False, tensorboard_log=False, save_results=True)
        if i == 3:
            exp_n_tops = [exp_n_tops[1]]
            test_candidates(filename=filename, candidate_list=exp_n_tops, dataset='cifar10', k=10, mode=mode, no_blocks=2, no_epochs=200, verbose=1, save_model=False, visualize=False, tensorboard_log=False, save_results=True)
        if i == 0:
            exp_n_tops = [exp_n_tops[1]]
            test_candidates(filename=filename, candidate_list=exp_n_tops, dataset='cifar10', k=10, mode=mode, no_blocks=2, no_epochs=200, verbose=1, save_model=False, visualize=False, tensorboard_log=False, save_results=True)

    test_benchmarks(dataset='cifar10', k=10, no_blocks=2, no_epochs=200, verbosity=0, save_model=False, visualize=False, tensorboard_log=False, save_results=True)
    #test_benchmarks(dataset='cifar100', k=10, no_blocks=2, no_epochs=200, verbosity=0, save_model=False, visualize=False, tensorboard_log=False, save_results=True)







if __name__ == "__main__":
    main()
