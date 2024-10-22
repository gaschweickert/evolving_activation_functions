from search import SEARCH

'''
RS class is used to initialize and run a "generation-based" random search. It is a subclass of search.
'''
class RS(SEARCH):
    def __init__(self, generations, N, C):
        super().__init__("RANDOM SEARCH", generations, N, C)

    # Runs search for the given number of generations using a desired architecture and training regminent
    def run(self, train_epochs, cnn, mode, number_of_blocks, verbosity):
        all_candidates = self.generate_n_unique_candidates(n=self.N*self.generations)
        for gen in range(1, self.generations + 1):
            print(self.search_type)
            self.population = all_candidates[(gen-1)*self.N : gen*self.N]
            for i, candidate in enumerate(self.population):
                print("\nGeneration #" + str(gen) + " : Candidate #" + str(i + 1))
                candidate.print_candidate_name()
                self.evaluate_candidate(candidate, train_epochs, cnn, mode, number_of_blocks, verbosity)
                candidate.print_candidate_results() # prints evaluated candidate results
                self.all_evaluated_candidate_solutions.append(candidate)
            print("\nGeneration #" + str(gen) + ' : Best Candidate')
            gen_best_candidate = self.get_population_best_candidate(evaluation_metric=2) # best accuracy wise
            gen_best_candidate.print_candidate_name_and_results()

  




