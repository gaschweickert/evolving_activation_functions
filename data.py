import csv
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
        for data in self.data:
            gen_max = []
            for gen in range(10):
                gen_data = data[gen*50: (gen + 1)*50]
                gen_best = max(gen_data, key=itemgetter(-1))
                gen_max.append(float(gen_best[-1]))
            xpoints = np.array(range(1, 11))
            ypoints = np.array(gen_max)
            plt.plot(xpoints, ypoints)


        plt.ylim((0.65, 0.8)) 

        plt.xlabel("Accuracy")
        plt.ylabel("Generation #")
        plt.title("ACCURACY vs GENERATIONS")

        plt.show()

    def print_overall_best(self):
        for data in self.data:
            print(max(data, key=itemgetter(-1)))
    
        




        




