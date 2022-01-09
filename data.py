import csv
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

class DATA:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.collect_data_from_file()
        self.header = []
        self.time_taken = []

    def collect_data_from_file(self):
        file = open(self.filename)
        csvreader = csv.reader(file)
        self.header = next(csvreader)
        rows = []
        for row in csvreader:
            rows.append(row)
        self.time_taken = rows[-1]
        return rows[:-1]

    def plot_gen_vs_accuracy(self):
        gen_max = []
        for gen in range(10):
            gen_data = self.data[gen*50: (gen + 1)*50]
            gen_best = max(gen_data, key=itemgetter(-1))
            gen_max.append(float(gen_best[-1]))


        xpoints = np.array(range(1, 11))
        ypoints = np.array(gen_max)

        plt.ylim((0.65, 0.8)) 

        plt.xlabel("Accuracy")
        plt.ylabel("Generation #")
        plt.title("ACCURACY vs GENERATIONS")

        plt.plot(xpoints, ypoints)
        plt.show()

    def print_overall_best(self):
        print(max(self.data, key=itemgetter(-1)))
    
        




        




