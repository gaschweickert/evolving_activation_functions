from genetic_algorithm import GA
from cnn_mnist import CNN
import csv

def main():

    save_file = True
    
    ga = GA()
    generations = 10
    k = 5 # number of folds for crossvalidation
    N = 50 # population size (>2 for crossover)
    C = 2 # search space complexity i.e. number of custom af (note: must change layer set up in CNN)
    m = 10 # number of new candidates per generation
    ga.initialize(N, C, m) 

    model = CNN()
    model.load_and_prep_data()

    gen_best_candidates = []
    for gen in range(1, generations + 1):
        for candidate_idx in range(N):
            print("\nGeneration #" + str(gen) + " : Candidate #" + str(candidate_idx))
            ga.print_candidate_name(ga.get_candidate(candidate_idx))
            evaluated_candidate = ga.evaluate_candidate(k, candidate_idx, model, custom=True, verbosity=0)
            ga.print_candidate_accuracy(evaluated_candidate)
        print("\nGeneration #" + str(gen) + ' : Best Candidate')
        gen_best_candidate = ga.get_population_best_candidate()
        ga.print_candidate_name_and_accuracy(gen_best_candidate)
        gen_best_candidates.append(gen_best_candidate)
        if (gen != generations): ga.evolve() # do not evolve final generation

    print("\nFinal best generated solution:")
    final_best_candidate = ga.get_population_best_candidate()
    ga.print_candidate_name_and_accuracy(final_best_candidate)

    relu_benchmark = ga.evaluate_candidate(k, None, model, custom=False)
    print("\nRelu benchmark accuracy:")
    print(relu_benchmark)

    # field names 
    fields = ['Gen', 'Candidate', 'Acc']


    if save_file:
        with open('CNN_MNIST_N=' + str(N) + '_C=' + str(C) + '_G=' + str(generations) + '_m=' + str(m), 'w') as f:
            
            # using csv.writer method from CSV package
            write = csv.writer(f)
            
            write.writerow(fields)
            for gen, candidate in enumerate(gen_best_candidates): 
                candidate_name = str(';'.join([cu.get_name() for cu in candidate[0]]))
                write.writerow([gen, candidate_name, candidate[1]])
            write.writerow([999, relu_benchmark[0], relu_benchmark[1]])





if __name__ == "__main__":
    main()
