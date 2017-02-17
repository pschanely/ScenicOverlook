import numbers
from scenicoverlook import viewablelist

#
# This is a trivial expression evaluator.
#
# Input should consist of single digit integers and two infix operators:
# addition (+) and multiplication (*).
#
# Incremental mapreduce allows us to efficiently update the result of very
# large arithmetic expressions when the input text changes.
#
# This also demonstrates generating viewablelists inside a mapreduce, which
# can effectively cascade the caching behavior over a chain (or even a
# graph) of operations.
#


PARSE_ERROR = 'PARSE_ERROR'

def combine_parses(left, right):
    if left is PARSE_ERROR or right is PARSE_ERROR:
        return PARSE_ERROR
    if left[-1] in ('*','+'):
        op = left[-1]
        left = left[:-1]
    elif right[0] in ('*','+'):
        op = right[0]
        right = right[1:]
    else:
        return PARSE_ERROR
    if left and right:
        if not isinstance(left[-1], numbers.Number):
            return PARSE_ERROR
        elif not isinstance(right[0], numbers.Number):
            return PARSE_ERROR
        if op == '*':
            return left[:-1] + [left[-1] * right[0]] + right[1:]
        elif op == '+':
            return left + right
        else:
            return PARSE_ERROR
    else: # left or right is empty; not enough data yet:
        return left + [op] + right
    

def eval(src):
    tokens = src.map(
        lambda ch: viewablelist([int(ch) if ch.isdigit() else ch]))
    products = tokens.reduce(combine_parses)
    if products is PARSE_ERROR:
        # One might think that if you break the parse, you would lose all the
        # cached values, but this is not so. Subsections of the input that
        # had valid parses continue to retain their product number lists,
        # and those cached product lists retain their sums for the reduce
        # below.
        return PARSE_ERROR
    else:
        return products.reduce(lambda x,y: x + y)

# Step 1: Start with an expression like this:
source = viewablelist('2*3*1')
print(' '.join(source) + ' => ' + str(eval(source)))
#           2 * 3 * 1   =>   6

# Step 2: Starting at the second character, type '1':
source = source[:2] + '1' + source[2:] 
print(' '.join(source) + ' => ' + str(eval(source)))
#       2 * 1   3 * 1   =>   PARSE_ERROR (you can't have two numbers together)

# Step 3: Then type '+':
source = source[:3] + '+' + source[3:]
print(' '.join(source) + ' => ' + str(eval(source)))
#       2 * 1 + 3 * 1   =>   5
