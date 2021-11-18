from genetic_algorithm import GA
from CNN_MNIST import CNN

def main():
    evolve = GA()
    evolve.population_initializer(N=1, C=2) # N is population size, C is solution complexity i.e. number of custom af
    print(evolve.get_population())

    for sol in evolve.get_encoded_population():
        print(sol)
        model = CNN()
        model.load_and_prep_data()
        model.set_custom_activation(sol)
        model.build_and_compile()
        model.train_and_validate()



if __name__ == "__main__":
    main()
