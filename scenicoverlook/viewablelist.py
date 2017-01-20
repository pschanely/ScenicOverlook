import functools
import itertools

_NO_VAL = object()


def viewablelist(rawlist=None):
    '''
    This implementation is adapted from this weight-based tree implementation
    in scheme:
    
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
        return ViewableList(_NO_VAL, None, None)
    def _helper(start, end):
        if start >= end:
            return None
        mid = (start + end) / 2
        return ViewableList(
            rawlist[mid], _helper(start, mid), _helper(mid + 1, end))
    return _helper(0, len(rawlist))

@functools.total_ordering
class ViewableList(object):
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
        if left:
            self._left = left
            self._count += self._left._count
        if right:
            self._right = right
            self._count += self._right._count
        if self._count > 1:
            # don't allocate a cache if we don't need it:
            self._reducevals = {}

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
                return viewablelist(self.to_list()[index])
            count = self._count
            start, end, _ = index.indices(count)
            cur = self
            if end < count:
                cur = _split_lt(cur, end)
            if start > 0:
                cur = _split_gte(cur, start)
            return viewablelist([]) if cur is None else cur
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
        return _concat2(self, other)

    def __repr__(self):
        return 'viewablelist({0})'.format(str(self.to_list()))

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        '''
        >>> list(viewablelist([]) + viewablelist([3]))
        [3]
        >>> range(100) == list(viewablelist(range(100)))
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
        return all(a == b for a, b in itertools.izip_longest(
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
        '''
        for a, b in itertools.izip(self.__iter__(), other.__iter__()):
            if a < b:
                return True
            elif b < a:
                return False
        return len(self) < len(other)

    def to_list(self):
        return list(self)

    def map_reduce(self, logic):
        reducer, mapper, initializer = logic
        ct = self._count
        if ct <= 1:
            return initializer if ct == 0 else mapper(self._val)
        else:
            rv = self._reducevals.get(logic, _NO_VAL)
            if rv is not _NO_VAL:
                return rv
            cur, left, right = self._val, self._left, self._right
            cur = mapper(cur)
            if left:
                cur = reducer(left.map_reduce(logic), cur)
            if right:
                cur = reducer(cur, right.map_reduce(logic))
            self._reducevals[logic] = cur
            return cur



        
def _single_l(av, x, r):
    bv, y, z = r._val, r._left, r._right
    return ViewableList(bv, ViewableList(av, x, y), z)

def _single_r(bv, l, z):
    av, x, y = l._val, l._left, l._right
    return ViewableList(av, x, ViewableList(bv, y, z))

def _double_l(av, x, r):
    cv, rl, z = r._val, r._left, r._right
    bv, y1, y2 = rl._val, rl._left, rl._right
    return ViewableList(bv,
                        ViewableList(av, x, y1),
                        ViewableList(cv, y2, z))

def _double_r(cv, l, z):
    av, x, lr = l._val, l._left, l._right
    bv, y1, y2 = lr._val, lr._left, lr._right
    return ViewableList(bv, 
                        ViewableList(av, x, y1),
                        ViewableList(cv, y2, z))

_TREE_RATIO = 5

def _tjoin(v, l, r):
    ln, rn = l._count if l else 0, r._count if r else 0
    if ln + rn < 2:
        return ViewableList(v, l, r)
    if rn > _TREE_RATIO * ln:
        # right is too big
        rl, rr = r._left, r._right
        rln = rl._count if rl else 0
        rrn = rr._count if rr else 0
        if rln < rrn:
            return _single_l(v, l, r)
        else:
            return _double_l(v, l, r)
    elif ln > _TREE_RATIO * rn:
        # left is too big
        ll, lr = l._left, l._right
        lln = ll._count if ll else 0
        lrn = lr._count if lr else 0
        if  lrn < lln:
            return _single_r(v, l, r)
        else:
            return _double_r(v, l, r)
    else:
        return ViewableList(v, l, r)

def _popmin(node):
    left, right = node._left, node._right
    if left is None:
        return (node._val, right)
    popped, tree = _popmin(left)
    return popped, _tjoin(node._val, tree, right)

def _concat2(node1, node2):
    if node1 is None or node1._count == 0:
        return node2
    if node2 is None or node2._count == 0:
        return node1
    min_val, node2 = _popmin(node2)
    return _tjoin(min_val, node1, node2)

def _concat3(v, l, r):
    if l is None or r is None:
        return _tjoin(v, l, r)
    else:
        n1, n2 = l._count, r._count
        if _TREE_RATIO * n1 < n2:
            v2, l2, r2 = r._val, r._left, r._right
            return _tjoin(v2, _concat3(v, l, l2), r2)
        elif _TREE_RATIO * n2 < n1:
            v1, l1, r1 = l._val, l._left, l._right
            return _tjoin(v1, l1, _concat3(v, r1, r))
        else:
            return ViewableList(v, l, r)

def _split_lt(node, x):
    if node is None or node._count == 0:
        return node
    left, right = node._left, node._right
    lc = left._count if left else 0
    if x < lc:
        return _split_lt(left, x)
    elif lc < x:
        return _concat3(node._val, left, _split_lt(right, x - (lc + 1)))
    else:
        return node._left

def _split_gte(node, x):
    if node is None or node._count == 0:
        return node
    left, right = node._left, node._right
    lc = left._count if left else 0
    if lc < x:
        return _split_gte(node._right, x - (lc + 1))
    elif x < lc:
        return _concat3(node._val, _split_gte(left, x), right)
    else:
        return _concat2(ViewableList(node._val, None, None), right)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
