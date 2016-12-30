import doctest

from collections import defaultdict, namedtuple


def _identity(x):
    return x

_NO_VAL = object()

class MapReduceLogic(object):
    __slots__ = ('reducer', 'mapper', 'initializer')
    def __init__(self, reducer=None, mapper=_identity, initializer=None):
        if reduce is None:
            raise ValueError('reducer may not be None')
        self.reducer = reducer
        self.mapper = mapper
        self.initializer = initializer
    def __iter__(self):
        return (self.reducer, self.mapper, self.initializer).__iter__()
    
class ViewableList(object):

    """
    This immutable list maintains the intermediate results of map/reduce 
    functions so that when the list is sliced or combined, the results of
    the map/reduce can be efficiently updated. (using previous results for
    parts of the original list(s) that remain unchanged)

    >>> mr = MapReduceLogic(reducer=lambda x, y: x + y, initializer=0)
    >>> l = ViewableList([1,2,1,2,1,2,1])
    >>> l.map_reduce(mr)
    10
    >>> l[2:].map_reduce(mr)
    7
    >>> l[:-1].map_reduce(mr)
    9
    >>> l[1:-2].map_reduce(mr)
    6

    Internally, sharing_list uses a binary tree to store its values. In
    each node, it caches results of the reduce function.

    Both the mapper and reducer function must be pure (they cannot modify
    anything, just return new values). Additionally, a reducer function, f,
    must be associative; that is:  f(f(x,y),z) == f(x, f(y,x))
    The reducer function need not be commutative (f(x,y) == f(y,x)), however.
    For instance, string concatenation is associative but not commutative;
    this example efficiently maintains a big string which depends on many
    small strings:

    >>> mr = MapReduceLogic(reducer=lambda x, y: x + ' ' + y, initializer='')
    >>> l = ViewableList(['the', 'quick', 'brown', 'fox'])
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
    >>> l = ViewableList([9, 3, 7, 5, 1])
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
    >>> l = ViewableList([9, 7, 3, 5, 1, 2, 4])
    >>> # window starts with first 4 elements and moves right:
    >>> l[0:4].map_reduce(mr)
    [7, 9]
    >>> l[1:5].map_reduce(mr)
    [5, 7]
    >>> l[2:6].map_reduce(mr)
    [3, 5]
    >>> l[3:7].map_reduce(mr)
    [4, 5]
    """
    
    __slots__ = ('_left', '_right', '_val', '_count','_reducevals')

    def __init__(self, values=None, pair=None):
        self._val = _NO_VAL
        self._reducevals = {}
        #self._logic = logic
        self._left = self._right = None
        if pair is not None:
            self._left, self._right = pair
            self._count = self._left._count + self._right._count
        if values is not None:
            if len(values) <= 1:
                if len(values) == 1:
                    self._val = values[0]
            else:
                mid = len(values) / 2
                self._left = ViewableList(values[:mid], None)
                self._right = ViewableList(values[mid:], None)
            self._count = len(values)

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None and index.step != 1:
                return ViewableList(self.tolist()[index], None)
            count = self._count
            start, end, _ = index.indices(count)
            if self._left is None:
                return self if end == 1 else ViewableList(None, None)
            if start == 0 and end == count:
                return self
            lcount = len(self._left)
            if lcount <= start:
                return self._right[start - lcount : end - lcount]
            elif lcount >= end:
                return self._left[start : end]
            else:
                return ViewableList(None, (self._left[start:], self._right[:end - lcount]))
        else:
            index = count + index if index < 0 else index
            left, right = self._left, self._right
            if left is not None:
                left_ct = len(left)
                return left[index] if index < left_ct else right[index - left_ct]
            elif index == 0:
                return self._val
            else:
                raise IndexError()
            
    def __add__(self, other):
        if not isinstance(other, ViewableList):
            other = ViewableList(other, None)
        return ViewableList(None, (self, other))

    def __repr__(self):
        return 'ViewableList({0})'.format(str(self.tolist()))

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        left, val, right = self._left, self._val, self._right
        for i in left:
            yield i
        yield val
        for i in right:
            yield i

    def __ne__(self, other):
        raise Exception()
        return compare_pvector(self, other, operator.ne)

    def __eq__(self, other):
        raise Exception()
        return self is other or compare_pvector(self, other, operator.eq)

    def __gt__(self, other):
        raise Exception()
        return compare_pvector(self, other, operator.gt)

    def __lt__(self, other):
        raise Exception()
        return compare_pvector(self, other, operator.lt)

    def __ge__(self, other):
        raise Exception()
        return compare_pvector(self, other, operator.ge)

    def __le__(self, other):
        raise Exception()
        return compare_pvector(self, other, operator.le)

    def __mul__(self, times):
        raise Exception()

    __rmul__ = __mul__

    def tolist(self):
        ret = []
        left, val, right = self._left, self._val, self._right
        if left is not None:
            ret.extend(left.tolist())
        if val is not _NO_VAL:
            ret.append(val)
        if right is not None:
            ret.extend(right.tolist())
        return ret

    def _totuple(self):
        return tuple(self.tolist())

    def __hash__(self):
        return hash(self._totuple())

    def map_reduce(self, logic):
        rv = self._reducevals.get(logic, _NO_VAL)
        if rv is not _NO_VAL:
            return rv
        reducer, mapper, initializer = logic
        if self._count <= 1:
            ret = mapper(self._val) if (self._count == 1) else initializer
        else:
            ret = reducer(self._left.map_reduce(logic),
                          self._right.map_reduce(logic))
        self._reducevals[logic] = ret
        return ret

    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
