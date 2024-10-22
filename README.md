## Evolutionary Optimization of activationfunctions in deep heterogeneous networks

Gregory Schweickert

Abstract: The effective use of deep neural networks often requires time consuming and expertisereliant manual design decisions. Existing meta-learning methods have come a long way toward alleviating these requirements and enabling end-to-end learning. However, one important yet often overlooked facet is the selection of activation functions. Although methods exist to optimize this process, the vast majority are only concerned with homogeneous networks. That is, optimizing a single activation function for all hidden units in a network. Conversely, heterogeneous networks employ a variety of activation functions throughout. In this work, an evolutionary search method for discovering well-performing homogeneous and heterogeneous activation function setups is presented. Using random search for baseline comparison, experiments show that the developed directed search method is well-suited for the task. Indeed, hand-engineered deep CNNs tested on CIFAR-10 using ReLU and Swish are outperformed by those using discovered solutions. Furthermore, the explored heterogeneous setups result in better performance than their homogeneous counterparts. Lastly, novel solutions of both types are shown to generalize to CIFAR-100. However, transfer to an up-scaled architecture is relatively less successful. The presented methods offer a promising new approach to meta-learning in the space of deep heterogeneous neural networks.

RUNNING ACTIVATION FUNCTION CONFIGURATION SEARCHES ON FIX-TOPOLOGY NETURAL NETWORKS
1. Choose between random and evolutionary search
2. Setup the experimental enviornment in main.py accordingly
3. Consider the following search parameters:
    - generations = generation cutoff point for searches
    - k = number of final test runs/repeats
    - N = population size (N-m-b>=2 for crossover)
    - C = search space complexity/degree of heterogeneity i.e. number of custom af (note: must match CNN arcchitecture)
    - m = number of new candidates per generation
    - b = number of preserved best candidates per generation
    - fitness_metric = 1 (loss) or 2 (accuracy) for fitness base metric
    - dataset = options are 'cifar10' or 'cifar100'
    - mode = 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    - number_of_blocks = number of VGG blocks (Note: number of layers = number of blocks * 2 + 1)
    - train_epochs = number of training epochs
4. By changing the above parameters the dataset, architecture, search settings, and search space can all be altered. Specifically, using the size, complexity, and mode parameters - homogeneous and heterogeneous search space can be explored.
5. Set save to True, if the results of the search should be logged

LOADING SEARCH DATA
1. The data generated from the searches i.e. every search candidate explored and their measure validation set performance, can be loaded with ease
2. Using methods from the DATA class, the candidate solutions can be regenerated for testing, exploration, and/or plotting purposes. 

TESTING 
1. In main.py, the test_candidates method calls upon a training/testing cycle which can be repeated any number of times. It also allows for the results to be logged for tensorboard or simply saved in a csv file. 
2. The test_benchmarks method does the same as in (1) but for ReLU and Swish.
3. There are a number of archiecture/model visualization methods implemented. These can be only be accessed in the testing phase. These can be accessed by changing the booleans visualize and/or save_model.





