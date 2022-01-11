import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

class DATA:
    def __init__(self):
        self.filenames = []
        self.data = []
        self.header = []
        self.time_taken = []

    def collect_data_from_file(self, filename):
        self.filenames.append(filename)
        file = open(filename)
        csvreader = csv.reader(file)
        self.header = next(csvreader)
        rows = []
        for row in csvreader:
            rows.append(row)
        self.time_taken.append(rows[-1])
        self.data.append(rows[:-1])

    def plot_gen_vs_accuracy(self):
        """
        generations = 10
        N = 50
        for i, data in enumerate(self.data):
            gens_max = []
            for gen in range(generations):
                gen_data = data[gen*N: (gen + 1)*N]
                nan_removed_gen_data = [x for x in gen_data if not math.isnan(float(x[-1]))]
                gen_max = max(nan_removed_gen_data, key=itemgetter(-1))
                gens_max.append(float(gen_max[-1]))
            print(gens_max)
            xpoints = np.array(range(1, 11))
            ypoints = np.array(gens_max)
            plt.plot(xpoints, ypoints, label=self.filenames[i][33:])
        """
        
        for i, data in enumerate(self.data):
            gens = []
            gens_best_accuracy = []
            prev_gen = 1
            gen_best_accuracy = 0.0 
            for entry in data:
                entry_gen = int(entry[0])
                if entry_gen != prev_gen:
                    gens.append(prev_gen)
                    gens_best_accuracy.append(gen_best_accuracy)
                    prev_gen = prev_gen + 1
                    gen_best_accuracy = 0.0            
                entry_accuracy = float(entry[-1])
                if not math.isnan(entry_accuracy):
                    if entry_accuracy > gen_best_accuracy:
                        gen_best_accuracy = entry_accuracy
            xpoints = np.array(gens)
            ypoints = np.array(gens_best_accuracy)
            plt.plot(xpoints, ypoints, label=self.filenames[i][33:])


        plt.ylim((0.65, 0.8)) 

        plt.xlabel("Accuracy")
        plt.ylabel("Generation #")
        plt.title("ACCURACY vs GENERATIONS")
        plt.legend()
        plt.show()

    def print_overall_best(self):
        for data in self.data:
            print(max(data, key=itemgetter(-1)))
        




        




