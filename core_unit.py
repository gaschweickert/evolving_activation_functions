
'''
The core unit class allows for the creation core unit objects. These represent
custom activation functions. They hold the keys describing their underlying elements
and can be evaluated directly through this object.
'''
class CORE_UNIT:
    def __init__(self, elementary_units_keys, elementary_units_functions):

        self.elementary_units_keys = elementary_units_keys
        self.elementary_units_functions = elementary_units_functions

    # A single string representing the AF in proper mathematical format is generated 
    def generate_name(self, elementary_units_keys):
        unary_u1, binary_u, unary_u2 = elementary_units_keys
        binary_u = binary_u.replace("x1", unary_u1)
        binary_u = binary_u.replace("x2", unary_u2)
        return binary_u

    # The generated name of the AF is printed
    def get_name(self):
        return self.generate_name(self.elementary_units_keys)

    # Returns the keys to the binary and unary units making up the core unit (AF)
    def get_elementary_units_keys(self):
        return list(self.elementary_units_keys)
    
    # Returns the binary and unary functions making up the core unit (AF)
    def get_elementary_units_functions(self):
        return self.elementary_units_functions

    # Used to set the keys to the binary and unary units making up the core unit (AF)
    def set_elementary_units_keys(self, elementary_units_keys):
        self.elementary_units_keys = elementary_units_keys

    # Used to set the binary and unary functions making up the core unit (AF)
    def set_elementary_units_functions(self, elementary_units_functions):
        self.elementary_units_functions = elementary_units_functions

    # Given a scalar x, the AF represented by the core unit is evaluated
    def evaluate_function(self, x):
        unary_u1, binary_u, unary_u2 =  self.elementary_units_functions
        unary_u1_x = unary_u1(x) if callable(unary_u1) else unary_u1
        unary_u2_x = unary_u2(x) if callable(unary_u2) else unary_u2
        return binary_u(unary_u1_x, unary_u2_x)
    
    # Verifies if the custom activationn functions is callable (takes a variable)
    def check_validity(self):
        unary_u1, binary_u, unary_u2 =  self.elementary_units_functions
        if not callable(unary_u1) and not callable(unary_u2):
            return False
        return True




    
