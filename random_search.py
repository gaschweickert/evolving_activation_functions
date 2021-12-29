from search import SEARCH

class RS(SEARCH):
    def __init__(self, generations, N, C):
        super().__init__("RANDOM SEARCH", generations, N, C)

    def run(self, k, train_epochs, cnn, mode, number_of_blocks):
        for gen in range(1, self.generations + 1):
            print(self.search_type)
            self.generate_N_candidates()
            self.print_population()
            for i, candidate in enumerate(self.population):
                print("\nGeneration #" + str(gen) + " : Candidate #" + str(i + 1))
                self.print_candidate_name(candidate)
                self.evaluate_candidate(candidate, k, train_epochs, cnn, mode, number_of_blocks, verbosity=0)
                self.print_candidate_results(candidate) # prints evaluated candidate results
                self.all_evaluated_candidate_solutions.append(candidate)
            print("\nGeneration #" + str(gen) + ' : Best Candidate')
            gen_best_candidate = self.get_population_best_candidate(evaluation_metric=2) # best accuracy wise
            self.print_candidate_name_and_results(gen_best_candidate)

  




