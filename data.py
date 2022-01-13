import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

class DATA:
    def __init__(self):
        self.filenames = []
        self.data = []
        self.nan_removed_data = []
        self.header = []
        self.time_taken = []

    def collect_data_from_file(self, filename):
        self.filenames.append(filename)
        file = open(filename)
        csvreader = csv.reader(file)
        self.header = next(csvreader)
        data = []
        nan_removed_data = []
        for row in csvreader:
            if not math.isnan(float(row[-1])):
                nan_removed_data.append(row)
            data.append(row)
        self.time_taken.append(data[-1])
        self.data.append(nan_removed_data[:-1])
        self.nan_removed_data.append(nan_removed_data[:-1])

    def plot_gen_vs_accuracy(self):
        for i, data in enumerate(self.nan_removed_data):
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

    def print_top_candidates(self):
        for i, data in enumerate(self.nan_removed_data):
            print(self.filenames[i])
            overall_best = max(data, key=itemgetter(-1))
            print(overall_best[1]) #candidate name
            print(overall_best[-1]) #accuracy

    def get_n_top_candidates(self, n):
        data_n_top = []
        for i, data in enumerate(self.nan_removed_data):
            sorted_data = sorted(data, key=itemgetter(-1))
            data_n_top.append([self.filenames[i], sorted_data[:n]])
        return data_n_top
        




        




