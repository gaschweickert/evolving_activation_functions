class CANDIDATE:
    def __init__(self, core_units, loss, accuracy):
        self.core_units = core_units
        self.loss = loss
        self.accuracy = accuracy

    def print_candidate_name_and_results(self):
        self.print_candidate_name()
        self.print_candidate_results()

    def print_candidate_name(self):
        print(self.get_candidate_name())

    def print_candidate_results(self):
        print('Loss: ' + str(self.loss) + '; Accuracy: ' + str(self.accuracy) + '\n')

    def check_candidate_validity(self):
        for core_unit in self.core_units:
            if not core_unit.check_validity():
                return False
        return True

    def get_candidate_name(self):
        if  isinstance(self.core_units, str): 
            return(self.core_units)
        else:
            for i in range(len(self.core_units)):
                return(self.core_units[i].get_name())

