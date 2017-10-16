""" Asserts or checks helper module """
def assert_equal(a, b, name_a=None, name_b=None, exc_type=ValueError):
    if not a == b:
        raise exc_type("var {0}:{1} not euqal to var {2}:{3}.".format(name_a, a, name_b, b))

def dim_check(lists, target_dim=None, names=None):
    if names is None:
        names = ['arg{:d}'.format(i) for i, _ in enumerate(lists)]
    assert_equal(len(lists), len(names), 'len of lists', 'len of names') 
    if target_dim is None:
        target_dim = len(lists[0])
    for l, n in zip(lists, names):
        if len(l) != target_dim:
            raise ValueError("Dim check failed with lists {0}, target {1} at {2}:{3}.".format(lists, target_dim, l, n))        
    return target_dim
    