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


    def plot_candidate(self, top_i, search_name, save=False):
        
        colors = ['b', 'g', 'r']
        for j, cu in enumerate(self.core_units):
            filename = cu.get_name().replace('/', chr(247))
            filename = filename.replace(' ', '')
            filename = filename.replace('*', chr(215))
            print('af_plots/'+ search_name[33:35] +'_top' + str(top_i+1) + '_custom' + str(j+1) + '_' + filename +'.png')

            xpoints = np.arange(-5, 5.01, 0.01)
            ypoints = []
            for x in xpoints:
                    ypoints.append(cu.evaluate_function(float(x)))
            plt.axhline(0,color='gray', linewidth=7, linestyle='--') # x = 0
            plt.axvline(0,color='gray', linewidth=7, linestyle='--') # y = 0
            plt.plot(xpoints, ypoints, color=colors[top_i], linewidth=9)
            plt.gca().spines[:].set_linewidth(7)
            plt.xticks([])
            plt.yticks([])

            plt.xlim(-5, 5)
            plt.tight_layout()
            
            if save: plt.savefig('af_plots/'+ search_name[33:35] +'_top' + str(top_i+1) + '_custom' + str(j+1) + '_' + filename +'.png')
            #plt.show()
            plt. clf() 

    

'''for x in xpoints:
                if j==1 and top_i == 0:
                    ypoints.append(None if x < 0 else cu.evaluate_function(float(x)))
                else:
                    ypoints.append(cu.evaluate_function(float(x)))
'''
