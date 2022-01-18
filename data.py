import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

from search import SEARCH

class DATA:
    def __init__(self):
        self.filenames = []
        self.data = []
        self.ordered_converted_data = []
        self.header = []
        self.time_taken = []


    def collect_data_from_file(self, filename):
        self.filenames.append(filename)
        file = open(filename)
        csvreader = csv.reader(file)
        self.header = next(csvreader)
        exp_data = []
        gen_data = []
        for entry in csvreader:
            if entry[0] == "Total time:":
                self.time_taken.append(entry[1])
                break
            entry_gen = int(entry[0])
            if entry_gen != (len(exp_data) + 1):
                exp_data.append(gen_data)
                gen_data = []
            gen_data.append(entry)
        exp_data.append(gen_data)
        self.data.append(exp_data)
    
    def convert_and_order(self):
        accuracy = lambda x: float(-0.0) if math.isnan(float(x[-1])) else float(x[-1])
        ss = SEARCH('None', 0,0,0)
        for exp_data in self.data:
            converted_exp_data = []
            for gen_data in exp_data:
                ordered_gen_data = sorted(gen_data, key=accuracy, reverse=True)
                converted_gen_data = []
                for entry in ordered_gen_data:
                    candidate_keys = [x for i, x in enumerate(entry[1:-2]) if i % 4]
                    C = len(candidate_keys)//3
                    candidate_keys=np.reshape(candidate_keys, (C, 3))
                    candidate = ss.generate_candidate_solution_from_keys(candidate_keys, loss=entry[-2], accuracy=entry[-1])
                    converted_gen_data.append(candidate)
                converted_exp_data.append(converted_gen_data)
            self.ordered_converted_data.append(converted_exp_data)

    def plot_gen_vs_accuracy(self):
        assert self.ordered_converted_data, "must convert and order data first"
        accuracy = lambda x: x.accuracy

        for i, exp_data in enumerate(self.ordered_converted_data):
            xpoints = np.array([gen for gen in range(1, len(exp_data) + 1)])
            ypoints = np.array([float(accuracy(gen[0])) for gen in exp_data])
            plt.plot(xpoints, ypoints, label=self.filenames[i][33:])

        plt.ylim((0.65, 0.8)) 
        plt.ylabel("Accuracy")
        plt.xlabel("Generation #")
        plt.title("ACCURACY vs GENERATIONS")
        plt.legend()
        plt.show()


    # unique
    def get_n_top_candidates(self, n, verbose=0):
        ss = SEARCH('None', 0,0,0)
        data_n_tops = []
        accuracy = lambda x: float(-0.0) if math.isnan(float(x.accuracy)) else float(x.accuracy)
        for i, exp_data in enumerate(self.ordered_converted_data):
            flat_candidate_list = [candidate for gen_data in exp_data for candidate in gen_data]
            ordered_flat_candidate_list = sorted(flat_candidate_list, key=accuracy, reverse=True)
            exp_top_n = [ordered_flat_candidate_list[0]]
            for can in ordered_flat_candidate_list:
                if len(exp_top_n) == n: break
                if all([not ss.check_same_candidate_solution(can, top_can) for top_can in exp_top_n]):
                    exp_top_n.append(can)
            data_n_tops.append(exp_top_n)
            if verbose: 
                print(self.filenames[i]) 
                for can in exp_top_n:
                    can.print_candidate_name_and_results()        
        return data_n_tops

    


    




        




