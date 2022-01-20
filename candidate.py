import matplotlib.pyplot as plt
import numpy as np

class CANDIDATE:
    def __init__(self, core_units, loss, accuracy):
        self.core_units = core_units
        self.loss = loss
        self.accuracy = accuracy

    def print_candidate_name_and_results(self):
        self.print_candidate_name()
        self.print_candidate_results()

    def print_candidate_name(self):
        for i, name in enumerate(self.get_candidate_name()):
            print("custom" + str(i + 1) + ": " + name)

    def print_candidate_results(self):
        print('Loss: ' + str(self.loss) + '; Accuracy: ' + str(self.accuracy) + '\n')

    def check_validity(self):
        for core_unit in self.core_units:
            if not core_unit.check_validity():
                return False
        return True

    def get_candidate_name(self):
        name = []
        if  isinstance(self.core_units, str): 
            name = self.core_units
        else:
            for i in range(len(self.core_units)):
                name.append(self.core_units[i].get_name())
        return name

    def get_candidate_complexity(self):
        return len(self.core_units)

    def enlarge(self, no_blocks):
        assert not no_blocks % (self.get_candidate_complexity() - 1), "Invalid transfer attempt, check solution complexity and cnn no_blocks"
        assert self.get_candidate_complexity() > 1, "Invalid transfer attempt, must be heterogeneous solution"
        new_core_units = []
        scaling_factor = no_blocks // (self.get_candidate_complexity() - 1)
        for cu in self.core_units[:-1]:
            for i in range(scaling_factor):
                new_core_units.append(cu)
        new_core_units.append(self.core_units[-1])
        self.core_units = new_core_units




    def plot_candidate(self):
        custom1 = self.core_units[0]
        xpoints = range(-5, 6)
        ypoints = []
        for x in xpoints:
            ypoints.append(custom1.evaluate_function(float(x)))
        plt.plot(xpoints, ypoints)
        plt.show()


