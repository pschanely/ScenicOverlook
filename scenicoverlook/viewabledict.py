from collections import Mapping
import itertools

_NO_VAL = object()


def viewabledict(rawdict=None):
    if not rawdict:
        return ViewableDict(_NO_VAL, _NO_VAL, None, None)
    all_kv = sorted(rawdict.iteritems())
    def _helper(start, end):
        if start >= end:
            return None
        mid = (start + end) / 2
        k, v = all_kv[mid]
        return ViewableDict(
            k, v, _helper(start, mid), _helper(mid + 1, end))
    return _helper(0, len(all_kv))


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
                cur = _split_gt(cur, start)
            if stop is not None:
                cur = _split_lt(cur, stop)
            return cur
        else:
            v = self.get(index, _NO_VAL)
            if v is _NO_VAL:
                raise KeyError(index)
            return v

    def get(self, index, default=None):
        key = self._key
        if index <= key:
            if index == key:
                return self._val
            left = self._left
            if left is None:
                return default
            else:
                return left.get(key, default)
        else:
            right = self._right
            if right is None:
                return default
            else:
                return right.get(key, default)

    def _depth(self):
        depth = 0 if self._key is _NO_VAL else 1
        l, r = self._left, self._right
        if l is not None:
            depth = max(depth, l._depth() + 1)
        if r is not None:
            depth = max(depth, r._depth() + 1)
        return depth
    
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
        >>> (viewabledict({0:0,1:1,2:2,3:3,4:4}) + viewabledict({5:5}) + viewabledict({6:6}))._depth()
        4
        '''
        if not isinstance(other, ViewableDict):
            other = viewabledict(other)
        return _union(self, other)

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
    
    def keys(self):
        '''
        >>> viewabledict({1:2, 2:3}).keys()
        [1, 2]
        '''
        left, key, right = self._left, self._key, self._right
        ret = []
        if left:
            ret.extend(left.keys())
        if key is not _NO_VAL:
            ret.append(key)
        if right:
            ret.extend(right.keys())
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
        return all(a == b for a, b in itertools.izip_longest(
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

    def map_reduce(self, logic):
        reducer, mapper, initializer = logic
        ct = self._count
        if ct <= 1:
            return initializer if ct == 0 else mapper(self._key, self._val)
        rv = self._reducevals.get(logic, _NO_VAL)
        if rv is not _NO_VAL:
            return rv
        ret = mapper(self._key, self._val)
        if self._left:
            ret = reducer(self._left.map_reduce(logic), ret)
        if self._right:
            ret = reducer(ret, self._right.map_reduce(logic))
        self._reducevals[logic] = ret
        return ret

        
def _single_l(ak, av, x, r):
    bk, bv, y, z = r._key, r._val, r._left, r._right
    return ViewableDict(bk, bv, ViewableDict(ak, av, x, y), z)

def _single_r(bk, bv, l, z):
    ak, av, x, y = l._key, l_val, l._left, l._right
    return ViewableDict(ak, av, x, ViewableDict(bk, bv, y, z))

def _double_l(ak, av, x, r):
    ck, cv, rl, z = r._key, r._val, r._left, r._right
    bk, bv, y1, y2 = rl._key, rl._val, rl._left, rl._right
    return ViewableDict(bk, bv,
                        ViewableDict(ak, av, x, y1),
                        ViewableDict(ck, cv, y2, z))

def _double_r(ck, cv, l, z):
    ak, av, x, lr = l._key, l._val, l._left, l._right
    bk, bv, y1, y2 = lr._key, lr._val, lr._left, lr._right
    return ViewableDict(bk, bv, 
                        ViewableDict(ak, av, x, y1),
                        ViewableDict(ck, cv, y2, z))

_TREE_RATIO = 5

def _tjoin(k, v, l, r):
    ln, rn = l._count if l else 0, r._count if r else 0
    if ln + rn < 2:
        return ViewableDict(k, v, l, r)
    if rn > _TREE_RATIO * ln:
        # right is too big
        rl, rr = r._left, r._right
        rln = rl._count if rl else 0
        rrn = rr._count if rr else 0
        if rln < rrn:
            return _single_l(k, v, l, r)
        else:
            return _double_l(k, v, l, r)
    elif ln > _TREE_RATIO * rn:
        # left is too big
        ll, lr = l._left, l._right
        lln = ll._count if ll else 0
        lrn = lr._count if lr else 0
        if  lrn < lln:
            return _single_r(k, v, l, r)
        else:
            return _double_r(k, v, l, r)
    else:
        return ViewableDict(k, v, l, r)

def _add(node, k, v):
    if node is None:
        return ViewableDict(k, v, None, None)
    key, val, l, r = node._key, node._val, node._left, node._right
    if  k < key:
        return _tjoin(key, val, _add(l, k, v), r)
    elif key < k:
        return _tjoin(key, val, l, _add(r, k, v))
    else:
        return ViewableDict(key, v, l, r)

def _concat3(k, v, l, r):
    if l is None:
        return _add(r, k, v)
    elif r is None:
        return _add(l, k, v)
    else:
        n1, n2 = l._count, r._count
        if _TREE_RATIO * n1 < n2:
            k2, v2, l2, r2 = r._key, r._val, r._left, r._right
            return _tjoin(k2, v2, _concat3(k, v, l, l2), r2)
        elif _TREE_RATIO * n2 < n1:
            k1, v1, l1, r1 = l._key, l._val, l._left, l._right
            return _tjoin(k1, v1, l1, _concat3(k, v, r1, r))
        else:
            return ViewableDict(k, v, l, r)

def _split_lt(node, x):
    if node is None or node._count == 0:
        return node
    k = node._key
    if x < k:
        return _split_lt(node._left, x)
    elif k < x:
        return _concat3(node._key, node._val, node._left, _split_lt(node._right, x))
    else:
        return node._left

def _split_gt(node, x):
    if node is None or node._count == 0:
        return node
    k = node._key
    if k < x:
        return _split_gt(node._right, x)
    elif x < k:
        return _concat3(node._key, node._val, _split_gt(node._left, x), node._right)
    else:
        return node._right

def _union(tree1, tree2):
    if tree1 is None or tree1._count == 0:
        return tree2
    if tree2 is None or tree2._count == 0:
        return tree1
    ak, av, l, r = tree2._key, tree2._val, tree2._left, tree2._right
    l1 = _split_lt(tree1, ak)
    r1 = _split_gt(tree1, ak)
    return _concat3(ak, av, _union(l1, l), _union(r1, r))

