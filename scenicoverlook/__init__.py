from viewablelist import viewablelist as _viewablelist
from viewabledict import viewabledict as _viewabledict
from mapreducelogic import MapReduceLogic

def viewablelist(rawlist=None):
    """
    This immutable list maintains the intermediate results of map/reduce 
    functions so that when the list is sliced or combined, the results of
    the map/reduce can be efficiently updated. (using previous results for
    parts of the original list that remain unchanged)

    >>> mr = MapReduceLogic(reducer=lambda x,y:x+y, initializer=0)
    >>> numbers = viewablelist([5, 10, 5, 0])
    >>> numbers.map_reduce(mr)
    20
    >>> numbers = numbers[1:]
    >>> numbers.map_reduce(mr)
    15
    >>> numbers = numbers + [10, 5]
    >>> numbers.map_reduce(mr)
    30

    Internally, viewablelist uses a binary tree to store its values. In
    each node, it caches results of the reduce function.

    Both the mapper and reducer function must be pure (they cannot modify
    anything, just return new values). Additionally, a reducer function, f,
    must be associative; that is:  f(f(x,y),z) == f(x, f(y,x))
    The reducer function need not be commutative (f(x,y) == f(y,x)), however.
    For instance, string concatenation is associative but not commutative;
    this example efficiently maintains a big string which depends on many
    small strings:

    >>> mr = MapReduceLogic(reducer=lambda x, y: x + ' ' + y, initializer='')
    >>> l = viewablelist(['the', 'quick', 'brown', 'fox'])
    >>> l.map_reduce(mr)
    'the quick brown fox'
    >>> (l[:2] + ['stealthy'] + l[2:]).map_reduce(mr)
    'the quick stealthy brown fox'
    
    In this example, we maintain a sorted view of a list using the map and
    reduce functions. Now we can make make arbitraty modifications to the
    original list and the view will update efficiently. (in linear time for
    most changes)

    >>> import heapq # (heapq.merge() sorts two already sorted lists)
    >>> mr = MapReduceLogic(reducer=lambda x, y: list(heapq.merge(x,y)),
    ...                     mapper=lambda x:[x], initializer=[])
    >>> l = viewablelist([9, 3, 7, 5, 1])
    >>> l.map_reduce(mr)
    [1, 3, 5, 7, 9]
    >>> (l + [ 4 ]).map_reduce(mr)
    [1, 3, 4, 5, 7, 9]
    >>> (l + l).map_reduce(mr)
    [1, 1, 3, 3, 5, 5, 7, 7, 9, 9]

    If instead, we only wanted the largest 2 values, we could simply add a
    [-2:] slice to the reduce function and be done with it. In this example,
    we efficiently maintain the largest two values over a rolling window of
    numbers:

    >>> import heapq
    >>> mr = MapReduceLogic(reducer=lambda x,y: list(heapq.merge(x,y))[-2:],
    ...                     mapper=lambda x:[x], initializer=[])
    >>> l = viewablelist([9, 7, 3, 5, 1, 2, 4])
    >>> # window starts with first 4 elements and moves right:
    >>> l[0:4].map_reduce(mr)
    [7, 9]
    >>> l[1:5].map_reduce(mr)
    [5, 7]
    >>> l[2:6].map_reduce(mr)
    [3, 5]
    >>> l[3:7].map_reduce(mr)
    [4, 5]

    Future Work:
    * New collection type: ViewableSet

    """
    return _viewablelist(rawlist)


def viewabledict(rawdict=None):
    '''
    This immutable dictionary maintains the intermediate results of map/reduce
    functions so that when the dictionary is sliced or combined, the results of
    the map/reduce can be efficiently updated. (using previous results for
    parts of the original dictionary that remain unchanged)

    TODO: more useful example here perhaps...
    >>> persons = viewabledict({'jim':23, 'sally':27, 'chiban':27})
    >>> mr = MapReduceLogic(
    ...   mapper=lambda k,v: {v:[k]},
    ...   reducer=lambda x,y:{k: x.get(k,[])+y.get(k,[]) for k in set(x.keys()+y.keys())})
    >>> persons.map_reduce(mr)
    {27: ['chiban', 'sally'], 23: ['jim']}
    >>> (persons + {'bill': 30}).map_reduce(mr)
    {27: ['chiban', 'sally'], 30: ['bill'], 23: ['jim']}

    Internally, viewabledict is implemented as a binary tree, ordered by its 
    keys. In each node, it caches results of the reduce function. The standard
    sequential getters over these dictionaries (keys(), iteritems(), etc) all
    iterate in key order.

    Unlike regular python dictionaries, sub-dictionaries can be sliced from
    viewabledicts using the python slice operator, "[:]".
    It is common and efficient to split these objects with slices, like so:

    >>> viewabledict({'alice': 1, 'jim': 2, 'sunny': 3})['betty':]
    viewabledict({'jim': 2, 'sunny': 3})
    >>> viewabledict({2:2, 3:3, 4:4})[:3]
    viewabledict({2: 2})

    Unlike list slices, both the minimum and maximum bounds are exclusive:

    >>> viewabledict({'alice':1, 'jim':2, 'sunny':3})['alice':'sunny'].keys()
    ['jim']

    viewabledicts can also be (immutably) combined with the plus operator:

    >>> viewabledict({1:1}) + viewabledict({2:2})
    viewabledict({1: 1, 2: 2})

    TODO: Need more immuable-dictionary helpers immutable set(), delete()

    More on map_reduce():

    Both the mapper and reducer function must be pure (they cannot modify
    anything, just return new values). Additionally, a reducer function, f,
    must be associative; that is:  f(f(x,y),z) == f(x, f(y,x))
    The reducer function need not be commutative (f(x,y) == f(y,x)), however.
    For instance, string concatenation is associative but not commutative;
    this example efficiently maintains a big string which is the 
    concatenation of dictionary values, in order of the numeric keys:

    >>> mr = MapReduceLogic(
    ...   mapper  =lambda k, v: v,
    ...   reducer =lambda x, y: x + ' ' + y)
    >>> l = viewabledict({3:'brown', 1:'the', 2:'quick', 4:'fox'})
    >>> l.map_reduce(mr)
    'the quick brown fox'
    >>> (l + {2.5: 'stealthy'}).map_reduce(mr)
    'the quick stealthy brown fox'
    
    '''
    return _viewabledict(rawdict)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
