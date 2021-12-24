from genetic_algorithm_search import GA
from cnn import CNN
import csv

def main():

    save_file = False
    
    generations = 5
    k = 2 # number of folds for crossvalidation
    N = 14 # population size (N-m-b>=2 for crossover)
    C = 1 # search space complexity i.e. number of custom af (note: must change layer set up in CNN)
    m = 2 # number of new candidates per generation
    b = 2 # number of preserved best candidates per generation
    evaluation_metric = 1 # 1 (loss) or 2 (accuracy) for fitness base metric
    dataset = "cifar10" # 'cifar10' or 'cifar100'
    mode = 1 # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    number_of_blocks = 1 # number of layers = number of blocks * 2 (+ 1)
    train_epochs = 5 #number of training epochs

    ga = GA(N, C, m, b)
    ga.initialize() 

    cnn = CNN()
    cnn.load_and_prep_data(dataset)

    gen_best_candidates = []
    for gen in range(1, generations + 1):
        for candidate_idx in range(N):
            print("\nGeneration #" + str(gen) + " : Candidate #" + str(candidate_idx + 1))
            ga.print_candidate_name(ga.get_candidate(candidate_idx))
            evaluated_candidate = ga.evaluate_candidate(candidate_idx, k, train_epochs, cnn, mode, number_of_blocks, verbosity=1)
            ga.print_candidate_results(evaluated_candidate)
        print("\nGeneration #" + str(gen) + ' : Best Candidate')
        gen_best_candidate = ga.get_population_best_candidate(evaluation_metric=2) # best accuracy wise
        ga.print_candidate_name_and_results(gen_best_candidate)
        gen_best_candidates.append(gen_best_candidate)
        #ga.print_population()
        if (gen != generations): ga.evolve(evaluation_metric) # do not evolve final generation

    print('Every generation best:')
    for i, can in enumerate(gen_best_candidates):
        print("\nGeneration #" + str(i + 1))
        ga.print_candidate_name_and_results(can)
    
    relu_benchmark = ga.evaluate_candidate(k, None, cnn, 0, number_of_blocks, verbosity=1)
    print("\nRelu benchmark accuracy:")
    print(relu_benchmark)


    #print("\nFinal best generated solution: (highest accuracy from all generations)")
    #final_best_candidate = max(ga.get_population_best_candidate(evaluation_metric))
    #ga.print_candidate_name_and_results(final_best_candidate)

    #relu_benchmark = ga.evaluate_candidate(k, None, cnn, mode, number_of_blocks, verbosity=1)
    #print("\nRelu benchmark accuracy:")
    #print(relu_benchmark)

    # field names 
    fields = ['Gen', 'Candidate', 'Loss', 'Acc']


    if save_file:
        with open('CNN_MNIST'+ '_' + dataset + '_N=' + str(N) + '_C=' + str(C) + '_G=' + str(generations) + '_m=' + str(m) + '_b=' + str(b), 'w') as f:
            
            # using csv.writer method from CSV package
            write = csv.writer(f)
            
            write.writerow(fields)
            for gen, candidate in enumerate(gen_best_candidates): 
                candidate_name = str(';'.join([cu.get_name() for cu in candidate[0]]))
                write.writerow([gen, candidate_name, candidate[1]])
            write.writerow([999, relu_benchmark[0], relu_benchmark[1]])





if __name__ == "__main__":
    main()
