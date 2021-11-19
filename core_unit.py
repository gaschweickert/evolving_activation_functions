


class CORE_UNIT:
    def __init__(self, elementary_units_names, elementary_units_functions):

        self.name = self.generate_name(elementary_units_names)
        self.elementary_units_functions = elementary_units_functions

    def generate_name(self, elementary_units_names):
        binary_u, unary_u1, unary_u2 = elementary_units_names
        binary_u = binary_u.replace("x1", unary_u1)
        binary_u = binary_u.replace("x2", unary_u2)
        return binary_u

    def get_name(self):
        return self.name

    def evaluate_function(self, x):
        binary_u, unary_u1, unary_u2 =  self.elementary_units_functions
        unary_u1_x = unary_u1(x) if callable(unary_u1) else unary_u1
        unary_u2_x = unary_u2(x) if callable(unary_u2) else unary_u2
        return binary_u(unary_u1_x, unary_u2_x)


    
