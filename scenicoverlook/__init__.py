from __future__ import absolute_import, division, print_function
from collections import Mapping
import itertools
import functools
import heapq

zip_longest = (itertools.izip_longest if hasattr(itertools, 'izip_longest')
               else itertools.zip_longest)
if hasattr(functools, 'reduce'):
    reduce = functools.reduce
    
#
# Public Interfaces
#

def viewablelist(rawlist=None):
    '''
    This immutable list maintains the intermediate results of map/reduce 
    functions so that when the list is sliced or combined, the results of
    the map/reduce can be efficiently updated. (using previous results for
    parts of the original list that remain unchanged)

    >>> adder = lambda l: l.reduce(lambda x,y:x+y, initializer=0)
    >>> numbers = viewablelist([5, 10, 5, 0])
    >>> adder(numbers)
    20
    >>> numbers = numbers[1:]
    >>> adder(numbers)
    15
    >>> numbers = numbers + [10, 5]
    >>> adder(numbers)
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

    >>> concat = lambda l: l.reduce(lambda x, y: x + ' ' + y, initializer='')
    >>> l = viewablelist(['the', 'quick', 'brown', 'fox'])
    >>> concat(l)
    'the quick brown fox'
    >>> concat(l[:2] + ['stealthy'] + l[2:])
    'the quick stealthy brown fox'
    
    In this example, we maintain a sorted view of a list using the map and
    reduce functions. Now we can make make arbitraty modifications to the
    original list and the view will update efficiently. (in linear time for
    most changes)

    >>> import heapq # (heapq.merge() sorts two already sorted lists)
    >>> def sorter(l):
    ...   l = l.map(lambda x:[x])
    ...   return l.reduce(lambda x, y: list(heapq.merge(x,y)),
    ...                   initializer=())
    >>> l = viewablelist([9, 3, 7, 5, 1])
    >>> sorter(l)
    [1, 3, 5, 7, 9]
    >>> sorter(l + [ 4 ])
    [1, 3, 4, 5, 7, 9]
    >>> sorter(l + l)
    [1, 1, 3, 3, 5, 5, 7, 7, 9, 9]

    If instead, we only wanted the largest 2 values, we could simply add a
    [-2:] slice to the reduce function and be done with it. In this example,
    we efficiently maintain the largest two values over a rolling window of
    numbers:

    >>> import heapq
    >>> def sorter(l):
    ...   l = l.map(lambda x:[x])
    ...   return l.reduce(lambda x, y: list(heapq.merge(x,y))[-2:],
    ...                   initializer=())
    >>> l = viewablelist([9, 7, 3, 5, 1, 2, 4])
    >>> # window starts with first 4 elements and moves right:
    >>> sorter(l[0:4])
    [7, 9]
    >>> sorter(l[1:5])
    [5, 7]
    >>> sorter(l[2:6])
    [3, 5]
    >>> sorter(l[3:7])
    [4, 5]

    The implementation of the backing tree (for both viewablelist and 
    viewabledict) is adapted from this weight-based tree implementation in
    scheme:
    
    ftp://ftp.cs.indiana.edu/pub/scheme-repository/code/struct/wttree.scm
    
    The tree should maintain an approximate balance under creation, slices and
    concatenation:
    
    >>> viewablelist(range(100))._depth()
    7
    >>> viewablelist(range(100))[40:50]._depth()
    4
    >>> l = reduce(lambda x,y:x+y, [viewablelist(range(10)) for _ in range(10)])
    >>> l._depth()
    9
    >>> l[40:50]._depth()
    4

    '''
    if not rawlist:
        return _EMPTY_LIST
    def _helper(start, end):
        if start >= end:
            return None
        mid = (start + end) // 2
        return ViewableList(
            rawlist[mid], _helper(start, mid), _helper(mid + 1, end))
    return _helper(0, len(rawlist))


def viewabledict(given=None):
    '''
    This immutable dictionary maintains the intermediate results of map/reduce
    functions so that when the dictionary is sliced or combined, the results of
    the map/reduce can be efficiently updated. (using previous results for
    parts of the original dictionary that remain unchanged)

    TODO: more useful example here perhaps...
    >>> persons = viewabledict({'jim':23, 'sally':27, 'chiban':27})
    >>> def by_age(p):
    ...   p = p.items().map(lambda kv: {kv[1]: [kv[0]]})
    ...   return p.reduce(lambda x, y:{k: x.get(k,[]) + y.get(k,[]) 
    ...                                for k in set(x.keys()).union(y.keys())})
    >>> by_age(persons)
    {27: ['chiban', 'sally'], 23: ['jim']}
    >>> by_age(persons + {'bill': 30})
    {27: ['chiban', 'sally'], 30: ['bill'], 23: ['jim']}

    Internally, viewabledict is implemented as a binary tree, ordered by its 
    keys. In each node, it caches results of the reduce function. The standard
    sequential getters over these dictionaries (keys(), iteritems(), etc) all
    iterate in key order. items(), keys(), and values() all return 
    viewablelists, which are frequently used to make further map/reduces.

    >>> vals = lambda l: l.values().reduce(lambda x, y: x + ' ' + y)
    >>> words = viewabledict({3:'brown', 1:'the', 2:'quick', 4:'fox'})
    >>> vals(words)
    'the quick brown fox'
    >>> vals(words + {2.5: 'stealthy'})
    'the quick stealthy brown fox'
    
    Unlike regular python dictionaries, sub-dictionaries can be sliced from
    viewabledicts using the python slice operator, "[:]".
    It is common and efficient to split these objects with slices, like so:

    >>> viewabledict({'alice': 1, 'jim': 2, 'sunny': 3})['betty':]
    viewabledict({'jim': 2, 'sunny': 3})
    >>> viewabledict({2:2, 3:3, 4:4})[:3]
    viewabledict({2: 2})

    Unlike list slices, both the minimum and maximum bounds are exclusive:

    >>> viewabledict({'alice':1, 'jim':2, 'sunny':3})['alice':'sunny'].keys()
    viewablelist(['jim'])

    viewabledicts can also be (immutably) combined with the plus operator, and
    you can make incremental new dictionaries with the set() and remove() 
    methods:

    >>> viewabledict({1: 1}) + viewabledict({2: 2})
    viewabledict({1: 1, 2: 2})
    >>> viewabledict({1: 1}).set(2, 2)
    viewabledict({1: 1, 2: 2})
    >>> viewabledict({1: 1}).remove(1)
    viewabledict({})

    '''
    if not given:
        return _EMPTY_DICT
    if isinstance(given, ViewableList):
        return _from_viewablelist(given)
    if isinstance(given, Mapping):
        given = given.items()
    all_kv = sorted(given)
    if len(all_kv) == 0:
        return _EMPTY_DICT
    def _helper(start, end):
        if start >= end:
            return None
        mid = (start + end) // 2
        k, v = all_kv[mid]
        return ViewableDict(
            k, v, _helper(start, mid), _helper(mid + 1, end))
    return _helper(0, len(all_kv))
    

def to_viewable(val, skip_types = ()):
    '''
    Recursively converts a structure of lists and dictionaries into
    their viewable variants.
    '''
    if isinstance(val, skip_types):
        return val
    if isinstance(val, (list, tuple)):
        return viewablelist([to_viewable(i, skip_types) for i in val])
    if isinstance(val, Mapping):
        return viewabledict({
            to_viewable(k, skip_types): to_viewable(v, skip_types) for
            (k, v) in val.items()
        })
    return val


def from_viewable(val):
    '''
    Recursively converts a structure of viewablelists and viewabledicts into
    lists and dictionaries.
    '''
    if isinstance(val, Mapping):
        return {from_viewable(k): from_viewable(v) for k,v in val.items()}
    elif isinstance(val, (ViewableList, MappedList, tuple, list)):
        return tuple(from_viewable(i) for i in val)
    else:
        return val

#
#  Implementation
#

def fnkey(fn):
    # Fancy footwork to get more precise equality checking on functions
    # By default, inner functions and lambda are newly created each
    # time they are evaluated.
    # However, often they do not actually use any of the variables in
    # their scope (the closure is empty). When this is the case, we can
    # consider them the same if their compiled bytecode is identical
    if fn is None:
        return None
    if fn.__closure__ is not None:
        return (fn.__code__, repr([cell.cell_contents for cell in fn.__closure__]))
    return fn

class MapReduceLogic(object):
    """
    Instances of this class simply hold the map and reduce functions,
    and for a defualt value when the reduce function does not have
    enough inputs. Different instances of this class are considered
    to be different (even if they contain the same functions), so be
    sure to create only one per use-case, and reuse it.
    """
    __slots__ = ('reducer', 'mapper', 'initializer')
    def __init__(self, reducer=None, mapper=None, initializer=None):
        self.reducer = reducer
        self.mapper = mapper
        self.initializer = initializer
    def __iter__(self):
        return (self.reducer, self.mapper, self.initializer).__iter__()
    def __eq__(self, other):
        r1, m1, i1 = self
        r2, m2, i2 = other
        return fnkey(r1) == fnkey(r2) and fnkey(m1) == fnkey(m2) and i1 == i2
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        r, m, i = self
        hsh = hash(fnkey(self.reducer))
        hsh = hsh * 31 + hash(fnkey(self.mapper))
        hsh = hsh * 31 + hash(self.initializer)
        return hsh

def _identity(x):
    return x

#
#  Implementation: viewablelist
#

class ViewableIterable(object):
    __slots__ = ()
    
    def map(self, fn):
        '''
        Applies a (pure) function to each element of this list. All maps are
        lazy; they will not be evaluated util the list is inspected or 
        manipulated in a more complex manner.
        >>> list(viewablelist([1,2,3]).map(lambda x:x+1))
        [2, 3, 4]
        >>> list(viewablelist([1,2,3]).map(lambda x:x+1).map(lambda x:x+1))
        [3, 4, 5]
        >>> viewablelist([1,2,3]).map(lambda x:x+1).reduce(lambda x,y:x+y)
        9
        '''
        return MappedList(_cached_partial(_wrap_fn_in_list, fn), self)
    
    def flatmap(self, fn):
        '''
        This transforms each member of the list into zero or more members,
        which are all spliced together into a resulting list.

        >>> list(viewablelist([2]).flatmap(lambda x:[x,x]))
        [2, 2]
        >>> list(viewablelist([2,3,4]).flatmap(lambda x:[]))
        []
        >>> list(viewablelist([1,2]).flatmap(lambda x:[x,x]))
        [1, 1, 2, 2]
        >>> list(viewablelist([]).flatmap(lambda x:[x,x]))
        []
        '''
        return MappedList(fn, self)

    def group_by(self, keyfn):
        '''
        Groups this list into sublists of items, I, for which keyfn(I) is
        equal. The return is a viewabledict that is keyed by keyfn(I). The
        values are viewablelists of those items which produce the 
        corresponding key. Within each group, the items retain the same
        ordering they had in the original list.

        For exmaple, this partitions a list into evens and odds:
        >>> viewablelist([2,5,3,0,9,11]).group_by(lambda x: x%2)
        viewabledict({0: viewablelist([2, 0]), 1: viewablelist([5, 3, 9, 11])})

        >>> viewablelist([2,5,2,2,5]).group_by(lambda x: x)
        viewabledict({2: viewablelist([2, 2, 2]), 5: viewablelist([5, 5])})

        '''
        ret = self.map(_cached_partial(_create_group, keyfn))
        return ret.reduce(_combine_groups)


@functools.total_ordering
class ViewableList(ViewableIterable):
    '''
    A tree-based implementation of a list that remembers previous 
    results of mapreduces and can re-use them for later runs.
    '''
    
    __slots__ = ('_left', '_right', '_val', '_count', '_reducevals')

    def __init__(self, val, left, right):
        self._left = None
        self._right = None
        self._count = 0
        self._val = val
        self._count = 0 if val is _NO_VAL else 1
        self._reducevals = {}
        if left:
            self._left = left
            self._count += self._left._count
        if right:
            self._right = right
            self._count += self._right._count

    def __len__(self):
        '''
        >>> len(viewablelist([]))
        0
        >>> len(viewablelist([1,2]))
        2
        >>> len(viewablelist([1,2]) + viewablelist([3]))
        3
        '''
        return self._count

    def __getitem__(self, index):
        '''
        >>> list(viewablelist([2, 3, 4])[:1])
        [2]
        >>> list(viewablelist([2, 3])[1:])
        [3]
        >>> list(viewablelist([2, 3])[2:])
        []
        >>> list(viewablelist([2, 3])[:-2])
        []
        >>> list(viewablelist([2,3,4,5])[2:])
        [4, 5]
        >>> list(viewablelist([9, 7, 3, 5, 1, 2, 4])[2:-1])
        [3, 5, 1, 2]
        >>> list(viewablelist(range(10))[::3])
        [0, 3, 6, 9]
        >>> viewablelist([2, 3, 4])[1]
        3
        >>> viewablelist([2, 3, 4])[-1]
        4
        '''
        if isinstance(index, slice):
            if index.step is not None and index.step != 1:
                return viewablelist(list(self)[index])
            count = self._count
            start, end, _ = index.indices(count)
            cur = self
            if end < count:
                cur = _lsplit_lt(cur, end)
            if start > 0:
                cur = _lsplit_gte(cur, start)
            return viewablelist() if cur is None else cur
        else:
            count, left, right = self._count, self._left, self._right
            index = count + index if index < 0 else index
            if left is not None:
                left_ct = left._count
                if index < left_ct:
                    return left[index]
                elif left_ct == index:
                    return self._val
                else:
                    return right[index - (left_ct + 1)]
            elif index == 0:
                return self._val
            else:
                return right[index - 1]

    def _depth(self):
        depth = 0 if self._val is _NO_VAL else 1
        l, r = self._left, self._right
        if l is not None:
            depth = max(depth, l._depth() + 1)
        if r is not None:
            depth = max(depth, r._depth() + 1)
        return depth
    
    def __add__(self, other):
        '''
        >>> (viewablelist([]) + viewablelist([3]))._depth()
        1
        >>> (viewablelist([0]) + viewablelist([1,2,3]))._depth()
        3
        >>> (viewablelist([0,1,2,3,4]) + viewablelist([5]))._depth()
        4
        >>> # (enough changes to trigger rebalance)
        >>> (viewablelist([0,1,2,3,4]) + viewablelist([5]) + 
        ...  viewablelist([6]))._depth()
        4
        '''
        if not isinstance(other, ViewableList):
            other = viewablelist(other)
        return _lconcat2(self, other)

    def __repr__(self):
        return 'viewablelist({0})'.format(str(list(self)))

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        '''
        >>> list(viewablelist([]) + viewablelist([3]))
        [3]
        >>> list(range(100)) == list(viewablelist(range(100)))
        True
        '''
        if self._count == 0:
            return
        cur, seconds = self, []
        while True:
            while cur is not None:
                seconds.append(cur)
                cur = cur._left
            if not seconds:
                return
            cur = seconds.pop()
            yield cur._val
            cur = cur._right

    def __ne__(self, other):
        return not self == other
    
    def __eq__(self, other):
        '''
        >>> viewablelist([]) != viewablelist([1])
        True
        >>> viewablelist([1,3]) != viewablelist([1,2])
        True
        >>> viewablelist([1,2]) + viewablelist([3]) == viewablelist([1,2,3])
        True
        >>> viewablelist([]) + viewablelist([3]) == viewablelist([3])
        True
        '''
        if not hasattr(other, '__iter__'):
            return False
        return all(a == b for a, b in zip_longest(
            self.__iter__(), other.__iter__(), fillvalue=_NO_VAL))

    def __hash__(self):
        '''
        >>> hash(viewablelist([1,2]) + viewablelist([3])) == hash(viewablelist([1,2,3]))
        True
        >>> hash(viewablelist([1])) != hash(viewablelist([2]))
        True
        '''
        hsh = 0
        for p in self:
            hsh = (hsh * 31) + hash(p)
        return hsh

    def __lt__(self, other):
        '''
        >>> viewablelist([]) < viewablelist([3])
        True
        >>> viewablelist([3]) < viewablelist([3])
        False
        >>> viewablelist([4]) < viewablelist([3, 4])
        False
        >>> # @functools.total_ordering gives us other comparison operators too:
        >>> viewablelist([3]) >= viewablelist([3])
        True
        >>> viewablelist([3]) >= viewablelist([3, 0])
        False
        >>> bool(viewablelist())
        False
        >>> bool(viewablelist([2]))
        True
        '''
        for a, b in zip(self.__iter__(), other.__iter__()):
            if a < b:
                return True
            elif b < a:
                return False
        return len(self) < len(other)

    def sorted(self, key=_identity):
        '''
        Sorts a list, optionally by a specified key function.
        If the key function returns an identical value for multiple list items,
        those items retian the order they had in the original list (the sort
        is stable).

        >>> viewablelist([4, 9, 0]).sorted()
        viewablelist([0, 4, 9])
        >>> viewablelist([4, 3, 1, 0, 2]).sorted(key=lambda x: x%2)
        viewablelist([4, 0, 2, 3, 1])
        >>> viewablelist([]).sorted()
        viewablelist([])

        '''
        reducer = _cached_partial(_merge_sorted_lists, key)
        return self.map(_wrap_in_list).reduce(reducer, initializer=_EMPTY_LIST)

    def filter(self, fn):
        flatfn = _cached_partial(_filter_to_flat, fn)
        return MappedList(flatfn, self)

    def reduce(self, reducer, initializer=None):
        mr = MapReduceLogic(mapper = _wrap_in_list,
                            reducer = reducer,
                            initializer = initializer)
        return self._map_reduce(mr)
    
    def _map_reduce(self, logic):
        rv = self._reducevals.get(logic, _NO_VAL)
        if rv is not _NO_VAL:
            return rv
        reducer, mapper, initializer = logic
        ct = self._count
        if ct == 0:
            return initializer
        c, left, right = self._val, self._left, self._right
        cur = mapper(c)
        if reducer:
            if len(cur) <= 1:
                cur = cur[0] if len(cur) == 1 else initializer
            else:
                cur = reduce(reducer, cur)
            if left:
                cur = reducer(left._map_reduce(logic), cur)
            if right:
                cur = reducer(cur, right._map_reduce(logic))
        else:
            ltree = left._map_reduce(logic) if left else None
            rtree = right._map_reduce(logic) if right else None
            if len(cur) == 1:
                cur = ViewableList(cur[0], ltree, rtree)
            else:
                cur = viewablelist(cur)
                if left:
                    cur = ltree + cur
                if right:
                    cur = cur + rtree
        self._reducevals[logic] = cur
        return cur


@functools.total_ordering
class MappedList(ViewableIterable):
    '''
    We implement maps lazily; this object represents a mapping function
    applied to a list.
    '''
    def __init__(self, fn, vl):
        self.inner = vl
        self.fn = fn
        self.realized = _NO_VAL
    def _realize(self):
        if self.realized is not _NO_VAL:
            return self.realized
        logic = MapReduceLogic(mapper=self.fn,
                               reducer=None,
                               initializer=_EMPTY_LIST)
        self.realized = self.inner._map_reduce(logic)
        return self.realized

    def __len__(self):
        return self._realize().__len__()
    def __getitem__(self, index):
        return self._realize().__getitem__(index)
    def __iter__(self):
        return self._realize().__iter__()
    def __add__(self, other):
        return self._realize().__add__(other)
    def __ne__(self, other):
        return self._realize().__new__(other)
    def __eq__(self, other):
        return self._realize().__eq__(other)
    def __lt__(self, other):
        return self._realize().__lt__(other)
    def sorted(self, key=_identity):
        return self._realize().sorted(key=key)
    def reduce(self, reducer, initializer=None):
        return self._map_reduce(MapReduceLogic(
            reducer=reducer, initializer=initializer))
    def _map_reduce(self, logic):
        fn = _cached_partial(_compose_flatmaps, (logic.mapper, self.fn))
        mr = MapReduceLogic(reducer=logic.reducer,
                            initializer=logic.initializer,
                            mapper=fn)
        return self.inner._map_reduce(mr)


    
#
#  Internal tree management functions for viewablelist
#



def _lsingle_l(av, x, r):
    bv, y, z = r._val, r._left, r._right
    return ViewableList(bv, ViewableList(av, x, y), z)

def _lsingle_r(bv, l, z):
    av, x, y = l._val, l._left, l._right
    return ViewableList(av, x, ViewableList(bv, y, z))

def _ldouble_l(av, x, r):
    cv, rl, z = r._val, r._left, r._right
    bv, y1, y2 = rl._val, rl._left, rl._right
    return ViewableList(bv,
                        ViewableList(av, x, y1),
                        ViewableList(cv, y2, z))

def _ldouble_r(cv, l, z):
    av, x, lr = l._val, l._left, l._right
    bv, y1, y2 = lr._val, lr._left, lr._right
    return ViewableList(bv, 
                        ViewableList(av, x, y1),
                        ViewableList(cv, y2, z))

_LTREE_RATIO = 5

def _ltjoin(v, l, r):
    ln, rn = l._count if l else 0, r._count if r else 0
    if ln + rn < 2:
        return ViewableList(v, l, r)
    if rn > _LTREE_RATIO * ln:
        # right is too big
        rl, rr = r._left, r._right
        rln = rl._count if rl else 0
        rrn = rr._count if rr else 0
        if rln < rrn:
            return _lsingle_l(v, l, r)
        else:
            return _ldouble_l(v, l, r)
    elif ln > _LTREE_RATIO * rn:
        # left is too big
        ll, lr = l._left, l._right
        lln = ll._count if ll else 0
        lrn = lr._count if lr else 0
        if  lrn < lln:
            return _lsingle_r(v, l, r)
        else:
            return _ldouble_r(v, l, r)
    else:
        return ViewableList(v, l, r)

def _lpopmin(node):
    left, right = node._left, node._right
    if left is None:
        return (node._val, right)
    popped, tree = _lpopmin(left)
    return popped, _ltjoin(node._val, tree, right)

def _lconcat2(node1, node2):
    if node1 is None or node1._count == 0:
        return node2
    if node2 is None or node2._count == 0:
        return node1
    min_val, node2 = _lpopmin(node2)
    return _ltjoin(min_val, node1, node2)

def _lconcat3(v, l, r):
    if l is None or r is None:
        return _ltjoin(v, l, r)
    else:
        n1, n2 = l._count, r._count
        if _LTREE_RATIO * n1 < n2:
            v2, l2, r2 = r._val, r._left, r._right
            return _ltjoin(v2, _lconcat3(v, l, l2), r2)
        elif _LTREE_RATIO * n2 < n1:
            v1, l1, r1 = l._val, l._left, l._right
            return _ltjoin(v1, l1, _lconcat3(v, r1, r))
        else:
            return ViewableList(v, l, r)

def _lsplit_lt(node, x):
    if node is None or node._count == 0:
        return node
    left, right = node._left, node._right
    lc = left._count if left else 0
    if x < lc:
        return _lsplit_lt(left, x)
    elif lc < x:
        return _lconcat3(node._val, left, _lsplit_lt(right, x - (lc + 1)))
    else:
        return node._left

def _lsplit_gte(node, x):
    if node is None or node._count == 0:
        return node
    left, right = node._left, node._right
    lc = left._count if left else 0
    if lc < x:
        return _lsplit_gte(node._right, x - (lc + 1))
    elif x < lc:
        return _lconcat3(node._val, _lsplit_gte(left, x), right)
    else:
        return _lconcat2(ViewableList(node._val, None, None), right)


    
#
#  Implementation: viewabledict
#



_CACHE_KEY_KEYS = object()
_CACHE_KEY_VALUES = object()
_CACHE_KEY_ITEMS = object()
_CACHE_KEY_FROM_LIST = object()

def _first(l):
    return l[0]

def _second(l):
    return l[1]

def _from_viewablelist(l):
    pair = l.val
    if l is _NO_VAL:
        return _EMPTY_DICT
    if len(pair) != 2:
        raise ValueError(
            'Dictionary must be given a list of pairs; {} is not a pair'
            .format(repr(pair)))
    cache = l._reducevals
    ret = cache.get(_CACHE_KEY_FROM_LIST)
    if ret:
        return ret
    left, right = l.left, l.right
    ret = ViewableDict(key, value,
                       left.to_viewabledict() if left else None,
                       right.to_viewable_dict() if right else None)
    cache[_CACHE_KEY_FROM_LIST] = ret
    return ret



class ViewableDict(Mapping):
    '''
    A tree-based implementation of a dictionary that remembers previous
    results of mapreduces and can re-use them for later runs.
    '''
    
    __slots__ = ('_left', '_right', '_key', '_val', '_count', '_reducevals')

    def __init__(self, key, val, left, right):
        self._left = None
        self._right = None
        self._count = 0
        self._key = key
        self._val = val
        self._count = 0 if key is _NO_VAL else 1
        self._reducevals = {}
        if left:
            self._left = left
            self._count += self._left._count
        if right:
            self._right = right
            self._count += self._right._count

    def __len__(self):
        '''
        >>> len(viewabledict({}))
        0
        >>> len(viewabledict({1:1,2:2}))
        2
        >>> len(viewabledict({1:1,2:2}) + viewabledict({3:3}))
        3
        '''
        return self._count

    def __getitem__(self, index):
        '''
        '''
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if step is not None:
                raise ValueError('Slices with steps are not supported')
            cur = self
            if start is not None:
                cur = _dsplit_gt(cur, start)
            if stop is not None:
                cur = _dsplit_lt(cur, stop)
            return cur
        else:
            v = self.get(index, _NO_VAL)
            if v is _NO_VAL:
                raise KeyError(index)
            return v

    def get(self, index, default=None):
        '''
        >>> l = viewabledict({1:1,2:2,3:3})
        >>> l[1], l[2], l[3]
        (1, 2, 3)
        >>> 9 in l, 1 in l
        (False, True)
        '''
        key = self._key
        if index <= key:
            if index == key:
                return self._val
            left = self._left
            if left is None:
                return default
            else:
                return left.get(index, default)
        else:
            right = self._right
            if right is None:
                return default
            else:
                return right.get(index, default)

    def _depth(self):
        depth = 0 if self._key is _NO_VAL else 1
        l, r = self._left, self._right
        if l is not None:
            depth = max(depth, l._depth() + 1)
        if r is not None:
            depth = max(depth, r._depth() + 1)
        return depth

    def set(self, key, val):
        '''
        Returns a new viewable dictionary with a given key set to a given value
        >>> viewabledict({3:3, 4:4}).set(3, 0)
        viewabledict({3: 0, 4: 4})
        '''
        return _dadd(self, key, val)
    
    def remove(self, key):
        '''
        Returns a new viewable dictionary with a given key removed.
        >>> viewabledict({3:3, 4:4}).remove(3)
        viewabledict({4: 4})
        >>> viewabledict({3:3, 4:4}).remove(99)
        viewabledict({3: 3, 4: 4})
        '''
        return (_dunion(_dsplit_lt(self, key), _dsplit_gt(self, key))
                or _EMPTY_DICT)
    
    def __add__(self, other):
        '''
        >>> viewabledict({3:3})._depth()
        1
        >>> (viewabledict({}) + viewabledict({3:3}))._depth()
        1
        >>> (viewabledict({0:0}) + viewabledict({1:1,2:2,3:3}))._depth()
        3
        >>> (viewabledict({0:0,1:1,2:2,3:3,4:4}) + viewabledict({5:5}))._depth()
        3
        >>> (viewabledict({0:0,1:1,2:2,3:3,4:4}) + viewabledict({5:5})
        ...  + viewabledict({6:6}))._depth()
        4
        '''
        if not isinstance(other, ViewableDict):
            other = viewabledict(other)
        return _dunion(self, other)

    def __repr__(self):
        return 'viewabledict({0})'.format(str(self.to_dict()))

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for k,v in self.iteritems():
            yield k

    def iteritems(self):
        '''
        >>> list(viewabledict({}).iteritems())
        []
        >>> list(viewabledict({1:1, 2:2}).iteritems())
        [(1, 1), (2, 2)]
        >>> list((viewabledict({1:1,2:2}) + viewabledict({3:3})).iteritems())
        [(1, 1), (2, 2), (3, 3)]
        >>> list(viewabledict({1:1,2:2,3:3}).iteritems())
        [(1, 1), (2, 2), (3, 3)]
        '''
        if self._count == 0:
            return
        cur, seconds = self, []
        while True:
            while cur is not None:
                seconds.append(cur)
                cur = cur._left
            if not seconds:
                return
            cur = seconds.pop()
            yield (cur._key, cur._val)
            cur = cur._right
    
    def items(self):
        '''
        >>> viewabledict({}).items()
        viewablelist([])
        >>> viewabledict({1:1, 2:2}).items()
        viewablelist([viewablelist([1, 1]), viewablelist([2, 2])])
        >>> (viewabledict({1:1,2:2}) + viewabledict({3:3})).items()
        viewablelist([viewablelist([1, 1]), viewablelist([2, 2]), viewablelist([3, 3])])
        >>> viewabledict({1:1,2:2,3:3}).items()
        viewablelist([viewablelist([1, 1]), viewablelist([2, 2]), viewablelist([3, 3])])
        '''
        if self._count == 0:
            return _EMPTY_LIST
        cache = self._reducevals
        ret = cache.get(_CACHE_KEY_ITEMS)
        if ret:
            return ret
        l, r = self._left, self._right
        ret = ViewableList(viewablelist([self._key, self._val]),
                           l.items() if l else None,
                           r.items() if r else None)
        cache[_CACHE_KEY_ITEMS] = ret
        return ret

    def keys(self):
        '''
        >>> viewabledict({1:2, 2:3}).keys()
        viewablelist([1, 2])
        '''
        if self._count == 0:
            return _EMPTY_LIST
        cache = self._reducevals
        ret = cache.get(_CACHE_KEY_KEYS)
        if ret:
            return ret
        l, r = self._left, self._right
        ret = ViewableList(self._key,
                           l.keys() if l else None,
                           r.keys() if r else None)
        cache[_CACHE_KEY_KEYS] = ret
        return ret

    def values(self):
        '''
        >>> viewabledict({1:2, 2:3}).values()
        viewablelist([2, 3])
        >>> viewabledict().values()
        viewablelist([])
        '''
        if self._count == 0:
            return _EMPTY_LIST
        cache = self._reducevals
        ret = cache.get(_CACHE_KEY_VALUES)
        if ret:
            return ret
        l, r = self._left, self._right
        ret = ViewableList(self._val,
                           l.values() if l else None,
                           r.values() if r else None)
        cache[_CACHE_KEY_VALUES] = ret
        return ret

    def memoized(self, fn):
        '''
        When using ViewableDictionary as a record or struct, you often want
        to process the whole object rather than reducing the key-value pairs.
        his function lets you transform the dictionary in whatever way you
        want, and caches the result so that it will not be recomputed if the
        dictionary hasn't changed.

        >>> d = viewabledict({'name':'Bob', 'age':20})
        >>> def compute_birth(person):
        ...   print('Computing birth year')
        ...   return person.set('birth_year', 2017 - person['age'])
        >>> sorted((k,v) for k,v in d.memoized(compute_birth).items())
        Computing birth year
        [('age', 20), ('birth_year', 1997), ('name', 'Bob')]
        >>> sorted((k,v) for k,v in d.memoized(compute_birth).items())
        [('age', 20), ('birth_year', 1997), ('name', 'Bob')]

        '''
        cache = self._reducevals
        fkey = fnkey(fn)
        if fkey in cache:
            return cache[fkey]
        ret = fn(self)
        cache[fkey] = ret
        return ret

    def __ne__(self, other):
        return not self == other
    
    def __eq__(self, other):
        '''
        >>> viewabledict({}) == viewabledict({})
        True
        >>> viewabledict({}) != viewabledict({1:1})
        True
        >>> viewabledict({1:1,3:3}) != viewabledict({1:1,2:2})
        True
        >>> viewabledict({1:1,2:2}) + viewabledict({3:3}) == viewabledict({1:1,2:2,3:3})
        True
        >>> viewabledict({}) + viewabledict({3:3}) == viewabledict({3:3})
        True
        '''
        if not isinstance(other, Mapping):
            return False
        return all(a == b for a, b in zip_longest(
            self.iteritems(), other.iteritems(), fillvalue=_NO_VAL))

    def __hash__(self):
        '''
        >>> hash(viewabledict({1:1,2:2}) + viewabledict({3:3})) == hash(viewabledict({1:1,2:2,3:3}))
        True
        >>> hash(viewabledict({1:1})) != hash(viewabledict({2:2}))
        True
        '''
        hsh = 0
        for p in self.iteritems():
            hsh = (hsh * 31) + hash(p)
        return hsh

    def to_dict(self):
        ret = {}
        def visit(node):
            left, key, val, right = node._left, node._key, node._val, node._right
            if left is not None:
                visit(left)
            if val is not _NO_VAL:
                ret[key] = val
            if right is not None:
                visit(right)
        visit(self)
        return ret

    def map_values(self, mapper):
        '''
        This can be used to efficiently transform just the values of a viewable 
        dictionary. This specialized method exists, because we can efficiently
        recreate the same tree structure when the keys are known not to change.
        The mapping function is given two arguments: the key and the values, 
        and is expected to return a transformed value.
        For other chainable transformations, use items(), keys(), or values(), 
        which all return viewable lists.

        >>> d = viewabledict({'chiban':7, 'bob':40})
        >>> d.map_values(lambda name, age: age + 1)
        viewabledict({'bob': 41, 'chiban': 8})
        
        '''
        mr = MapReduceLogic(mapper=mapper,
                            reducer=None)
        return self._map_reduce(mr)
    
    def _map_reduce(self, logic):
        rv = self._reducevals.get(logic, _NO_VAL)
        if rv is not _NO_VAL:
            return rv
        reducer, mapper, initializer = logic
        ct = self._count
        if ct == 0:
            return initializer
        key, val, left, right = self._key, self._val, self._left, self._right
        newval = mapper(key, val)
        if reducer is None:
            ltree = left._map_reduce(logic) if left else None
            rtree = right._map_reduce(logic) if right else None
            if newval is val and ltree is left and rtree is right:
                cur = self # nothing has changed, return ourself
            else:
                cur = ViewableDict(key, newval, ltree, rtree)
        else:
            if left:
                cur = reducer(left._map_reduce(logic), newval)
            if right:
                cur = reducer(newval, right._map_reduce(logic))
        self._reducevals[logic] = cur
        return cur


    
#
#  Internal tree management functions for viewabledict
#



def _dsingle_l(ak, av, x, r):
    bk, bv, y, z = r._key, r._val, r._left, r._right
    return ViewableDict(bk, bv, ViewableDict(ak, av, x, y), z)

def _dsingle_r(bk, bv, l, z):
    ak, av, x, y = l._key, l._val, l._left, l._right
    return ViewableDict(ak, av, x, ViewableDict(bk, bv, y, z))

def _ddouble_l(ak, av, x, r):
    ck, cv, rl, z = r._key, r._val, r._left, r._right
    bk, bv, y1, y2 = rl._key, rl._val, rl._left, rl._right
    return ViewableDict(bk, bv,
                        ViewableDict(ak, av, x, y1),
                        ViewableDict(ck, cv, y2, z))

def _ddouble_r(ck, cv, l, z):
    ak, av, x, lr = l._key, l._val, l._left, l._right
    bk, bv, y1, y2 = lr._key, lr._val, lr._left, lr._right
    return ViewableDict(bk, bv, 
                        ViewableDict(ak, av, x, y1),
                        ViewableDict(ck, cv, y2, z))

_DTREE_RATIO = 5

def _dtjoin(k, v, l, r):
    ln, rn = l._count if l else 0, r._count if r else 0
    if ln + rn < 2:
        return ViewableDict(k, v, l, r)
    if rn > _DTREE_RATIO * ln:
        # right is too big
        rl, rr = r._left, r._right
        rln = rl._count if rl else 0
        rrn = rr._count if rr else 0
        if rln < rrn:
            return _dsingle_l(k, v, l, r)
        else:
            return _ddouble_l(k, v, l, r)
    elif ln > _DTREE_RATIO * rn:
        # left is too big
        ll, lr = l._left, l._right
        lln = ll._count if ll else 0
        lrn = lr._count if lr else 0
        if  lrn < lln:
            return _dsingle_r(k, v, l, r)
        else:
            return _ddouble_r(k, v, l, r)
    else:
        return ViewableDict(k, v, l, r)

def _dadd(node, k, v):
    if node is None:
        return ViewableDict(k, v, None, None)
    key, val, l, r = node._key, node._val, node._left, node._right
    if  k < key:
        return _dtjoin(key, val, _dadd(l, k, v), r)
    elif key < k:
        return _dtjoin(key, val, l, _dadd(r, k, v))
    else:
        return ViewableDict(key, v, l, r)

def _dconcat3(k, v, l, r):
    if l is None:
        return _dadd(r, k, v)
    elif r is None:
        return _dadd(l, k, v)
    else:
        n1, n2 = l._count, r._count
        if _DTREE_RATIO * n1 < n2:
            k2, v2, l2, r2 = r._key, r._val, r._left, r._right
            return _tjoin(k2, v2, _dconcat3(k, v, l, l2), r2)
        elif _DTREE_RATIO * n2 < n1:
            k1, v1, l1, r1 = l._key, l._val, l._left, l._right
            return _dtjoin(k1, v1, l1, _dconcat3(k, v, r1, r))
        else:
            return ViewableDict(k, v, l, r)

def _dsplit_lt(node, x):
    if node is None or node._count == 0:
        return node
    k = node._key
    if x < k:
        return _dsplit_lt(node._left, x)
    elif k < x:
        return _dconcat3(node._key, node._val, node._left, _dsplit_lt(node._right, x))
    else:
        return node._left

def _dsplit_gt(node, x):
    if node is None or node._count == 0:
        return node
    k = node._key
    if k < x:
        return _dsplit_gt(node._right, x)
    elif x < k:
        return _dconcat3(node._key, node._val, _dsplit_gt(node._left, x), node._right)
    else:
        return node._right

def _dunion(tree1, tree2):
    if tree1 is None or tree1._count == 0:
        return tree2
    if tree2 is None or tree2._count == 0:
        return tree1
    ak, av, l, r = tree2._key, tree2._val, tree2._left, tree2._right
    l1 = _dsplit_lt(tree1, ak)
    r1 = _dsplit_gt(tree1, ak)
    return _dconcat3(ak, av, _dunion(l1, l), _dunion(r1, r))



#
#  _cached_partial and utilities
#


_PARTIALS_CACHE = {}

def _cached_partial(fn, value_to_apply):
    '''
    This is used to wrap functions in a way that preserves identity of
    the resulting function objects. For example,

    >>> first = lambda l: l[0]
    >>> sorter_by = lambda keyfn : (lambda l: sorted(l, key=keyfn))
    >>> sorter_by(first) == sorter_by(first)
    False
    >>> _cached_partial(sorter_by, first) == _cached_partial(sorter_by, first)
    True
    '''
    cache_key = (fn, value_to_apply)
    ret = _PARTIALS_CACHE.get(cache_key)
    if not ret:
        ret = fn(value_to_apply)
        _PARTIALS_CACHE[cache_key] = ret
    return ret

def _l(*items):
    return viewablelist(items)

def _wrap_in_list(v):
    return ViewableList(v, None, None)

def _filter_to_flat(fn):
    return lambda x: _wrap_in_list(x) if fn(x) else ()

def _compose_flatmaps(fns):
    f1, f2 = fns
    if f1 is _wrap_in_list or f1 is None: return f2
    if f2 is _wrap_in_list or f2 is None: return f1
    return lambda v: viewablelist([v1 for v2 in f2(v) for v1 in f1(v2)])

def _wrap_fn_in_list(fn):
    return lambda x: _wrap_in_list(fn(x))

def _mergesorted(i1, i2, keyfn=_identity):
    '''
    >>> list(_mergesorted(iter([]), iter([])))
    []
    >>> list(_mergesorted(iter([5]), iter([])))
    [5]
    >>> list(_mergesorted(iter([]), iter([5])))
    [5]
    >>> list(_mergesorted(iter([1,5]), iter([])))
    [1, 5]
    >>> list(_mergesorted(iter([]), iter([1,5])))
    [1, 5]
    >>> list(_mergesorted(iter([1,5]), iter([3])))
    [1, 3, 5]
    >>> list(_mergesorted(iter([9]), iter([1,5])))
    [1, 5, 9]
    >>> list(_mergesorted(iter([0]), iter([1,5])))
    [0, 1, 5]
    '''
    v1, v2, _StopIteration = _NO_VAL, _NO_VAL, StopIteration
    try:
        v1 = next(i1)
    except _StopIteration:
        pass
    try:
        v2 = next(i2)
        if v1 is not _NO_VAL:
            while True:
                if keyfn(v1) <= keyfn(v2):
                    yield v1
                    try:
                        v1 = next(i1)
                    except _StopIteration:
                        v1 = _NO_VAL
                        break
                else:
                    yield v2
                    try:
                        v2 = next(i2)
                    except _StopIteration:
                        v2 = _NO_VAL
                        break
    except _StopIteration:
        pass
    if v1 is not _NO_VAL:
        yield v1
        for v in i1:
            yield v
    if v2 is not _NO_VAL:
        yield v2
        for v in i2:
            yield v
        
def _create_group(keyfn):
    return lambda val: viewabledict({keyfn(val): _l(val)})

def _combine_groups(groups1, groups2):
    prev_key, prev_values = _NO_VAL, None
    result = []
    for (key, values) in _mergesorted(groups1.iteritems(), groups2.iteritems(), _first):
        if key == prev_key:
            values = prev_values + values
        elif prev_key is not _NO_VAL:
            result.append((prev_key, prev_values))
        prev_key, prev_values = key, values
    # add the last group
    if prev_key is not _NO_VAL:
        result.append((prev_key, prev_values))
    return viewabledict(result)
    
def _merge_sorted_lists(keyfn):
    def merge(l1, l2):
        # TODO might be nice to have a specialized version of this which
        # re-uses uninterrupted subtrees from one side; it would boost
        # lists that are mostly sorted already.
        return viewablelist(list(_mergesorted(iter(l1), iter(l2), keyfn)))
    return merge



#
#  Other
#



_NO_VAL = object()

_EMPTY_LIST = ViewableList(_NO_VAL, None, None)

_EMPTY_DICT = ViewableDict(_NO_VAL, _NO_VAL, None, None)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
