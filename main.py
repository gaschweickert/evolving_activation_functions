from genetic_algorithm import GA
from cnn_mnist import CNN

def main():
    
    ga = GA()
    ga.initialize_population(N=10, C=2) # N is population size, C is solution complexity i.e. number of custom af (must change layer set up in CNN)

    model = CNN()
    model.load_and_prep_data()
    
    for gen in generations:
        candidates_fitness = []
        for i, candidate in N: #N times
            ga.evaluate(i, model)
        ga.update_fitness(candidates_fitness)
        
        



    model = CNN()
    model.load_and_prep_data()
    for i, sol in enumerate(evolve.get_population()):
        model.set_custom_activation(sol)
        print('Candidate Solution #', i)
        for j, af in enumerate(sol):
            print(' Custom AF' + str(j) + ': ' + af.get_name())
        model.build_and_compile()
        model.summary()
        model.train_and_validate()
        results = model.evaluate()
        print("test loss, test acc:", results)




if __name__ == "__main__":
    main()
