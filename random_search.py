from search import SEARCH

class RS(SEARCH):
    def __init__(self, generations, N, C):
        super().__init__("RANDOM SEARCH", generations, N, C)

    def run(self, k, train_epochs, cnn, mode, number_of_blocks):
        gen_best_candidates = []
        for gen in range(1, self.generations + 1):
            print(self.search_type)
            self.generate_N_candidates()
            for candidate_idx in range(self.N):
                print("\nGeneration #" + str(gen) + " : Candidate #" + str(candidate_idx + 1))
                self.print_candidate_name(self.get_candidate(candidate_idx))
                evaluated_candidate = self.evaluate_candidate(candidate_idx, k, train_epochs, cnn, mode, number_of_blocks, verbosity=1)
                self.print_candidate_results(evaluated_candidate)
            print("\nGeneration #" + str(gen) + " : Best Candidate")
            gen_best_candidate = self.get_population_best_candidate(evaluation_metric=2) # best accuracy wise
            self.print_candidate_name_and_results(gen_best_candidate)
            gen_best_candidates.append(gen_best_candidate)
            #self.print_population()


