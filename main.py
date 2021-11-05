from genetic_algorithm import GA

def main():
    evolve = GA()
    evolve.population_initializer(10, 1)
    print(evolve.get_population())

if __name__ == "__main__":
    main()
