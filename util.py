
def apply_op(a, op, b):
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq,
           '=': operator.eq,
           '&': operator.and_}
    return ops[op](a, b)
