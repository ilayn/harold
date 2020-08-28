# ----------------------------------
# Author: Samuel Law
# Date 8/23/2020
#
# Bugs:
# Todo: Fix the subs function
# ---------------------------------


import sympy as sym
from itertools import chain
from math import floor


class RouthArray():
    """Small class that generates a
    Routh Array and provides easy to
    use printing and analysis tools."""

    def __init__(self, characteristic_equation, var):
        """Computes a routh array and stores it, due to the
        __str__ method being over-ridden, you can call print
        on the array upon declaration if all that is needed
        is to output the array to the console.
        args:
            charcteristic_equation:
                sympy sym, or expr object
            var:
                polynomial symbol, for example "x" is x**2 + 5
        attr:
            array:
                routh array table
            print_rounding:
                decimal places to round to for printing
            degree:
                degree of ce
            var:
                polynomial variable
        """

        # settings
        self.print_rounding = 3
        self.print_width = 100

        # calculations
        # capture attrs for internal use
        self.array, self.degree, self.var = self._compute_array(
            characteristic_equation, var
        )

    def __call__(self, row_index, col_index):
        """allows the RouthArray object to be used
        as a "functor" after instantiation to grab a
        specific index from the array """
        return self.array[row_index, col_index]

    def __str__(self):
        """Returns  a sympy matrix to
        print so everything looks super pretty."""

        # make sure everything is simplified
        tbl = [self.array.row(n) for n in range(self.array.rows)]
        tbl = [[sym.simplify(item) for item in row] for row in tbl]
        tbl = [[str(self.round_expr(c, self.print_rounding)) for c in row] for row in tbl]

        # maxlen = ceil(self.print_width/ceil(((self.degree + 1)/2))) - len(tbl[0])
        maxlen = floor((self.print_width - len(tbl[0]))/len(tbl[0]))
        longest = max([len(c) for c in chain(*tbl)])

        length = longest if longest < maxlen else maxlen

        def wrap(x):
            result = []
            nonlocal length

            # slice the list into chuncks
            while len(x) > length:
                idx = x[:length].rfind(" ")
                if idx > 0:
                    idx += 1
                    result.append(x[:idx])
                    x = x[idx:]
                else:
                    result.append(x[:length])
                    x = x[length:]

            # ensure complete str capture
            result.append(x[:])

            # fill in empty spots with spaces
            for i, item in enumerate(result):
                if len(item) < length:
                    item = ''.join([item, ' '*(length - len(item))])
                    result[i] = item

            # return the results
            return result

        tbl = [[wrap(c) for c in row] for row in tbl]

        def merge(row):
            nonlocal length
            line = ""
            for lnum in range(max([len(item) for item in row])):
                for item in row:
                    try:
                        line += item[lnum]
                    except IndexError:
                        line += " "*length
                    finally:
                        line += '|'
                line += '\n'
            return line

        tbl = [merge(row) for row in tbl]
        template = '-'*max([len(x) for x in tbl[0].splitlines()]) + '\n'
        tbl = template.join(tbl)

        return tbl

    def __repr__(self):
        return self.__str__()

    # --------helper functions---------

    def format_poly(self, poly, var):
        """Formats a polynomial to allow
        for generation of symbolic routh array."""

        poly = sym.simplify(poly)
        poly = sym.expand(poly)
        poly = sym.simplify(poly)
        poly = sym.collect(poly, var)
        return poly

    def get_coeffs(self, formatted_poly, var):
        """Returns the coefficients of a
        formated_polynomialwith respect
        to the variable var."""

        p = sym.Poly(formatted_poly, var)
        c = p.all_coeffs()
        return c

    def round_expr(self, expr, num_digits):
        try:
            val = expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sym.Number)})
        except AttributeError:
            val = expr
        finally:
            return val

    def set_print_width(self, width_in_chars: int):
        """This function is just for clarity, feel
        free to change the print width attribute directly"""

        self.print_width = width_in_chars

    def get_print_width(self):
        """This function is just for clarity, feel
        free to print() the print width attribute directly"""

        return self.print_width


    # --------general functions---------

    def subs(self, replacement_dict):
        self.array = self.array.subs(replacement_dict)

    def _compute_array(self, characteristic_equation, var):
        """computes routh array given a
        characteristic equation
        args:
            characteristic_equation: sympy expr or poly object
                such as x**2 + 8*x + (15 - K)
            var: symbol that defines poly, shown as "x" in the
                example above
        returns:
            (array, deg, var)
        """

        # extract the coefficients
        poly = self.format_poly(characteristic_equation, var)
        coeffs = self.get_coeffs(poly, var)

        # start the first two rows of the  table
        # using the alternating pattern
        tbl = [[], []]
        tbl[0] = [c for c in coeffs[::2]]
        tbl[1] = [c for c in coeffs[1::2]]

        # append zero if the second row is shorter
        # than the first
        if (len(tbl[0]) != len(tbl[1])) and (len(coeffs) > 1):
            tbl[1].append(0)

        # fill up the rest of the routh array, starting at the third row
        for r in range(2, len(coeffs)):
            tbl.append([])

            # start enumeration on second row
            for c, _ in enumerate(tbl[r-1][:-1]):
                try:
                    if tbl[r-1][0] == 0:  # ensure zero errors are caught
                        raise ZeroDivisionError
                    temp = (tbl[r-1][0] * tbl[r-2][c+1]) - \
                           (tbl[r-2][0] * tbl[r-1][c+1])
                    temp = temp/tbl[r-1][0]
                except ZeroDivisionError:
                    temp = sym.nan
                finally:
                    tbl[r].append(temp)

            # fill in additional zeros
            while len(tbl[r]) != len(tbl[0]):
                tbl[r].append(0)

        return (sym.Matrix(tbl), len(coeffs)-1, var)

    @property
    def domain(self):
        """Using the first column of the array
        it determines the domain assuming entry is
        univariable and the desired value of the cell
        to to be greater than zero

        Raises not implimented error if there is more than one
        variable of interest"""

        return sym.solve([e > 0 for e in self.array[:, 0]])

        # function adapted from the below link
        # https://dynamics-and-control.readthedocs.io/en/latest/2_Control/2_Laplace_domain_analysis_of_control_systems/SymPy%20Routh%20Array.html"""
