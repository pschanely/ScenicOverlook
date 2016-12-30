===============
Scenic Overlook
===============

The Scenic Overlook library contains datastructures for incremental
map-reduces.

These datastructures are implemented as trees, and store at each
node, intermediate values of the reduce. This means that when you
slice or combine structures, the new output of the map-reduce can
be efficiently computed. (by reusing old outputs from unchanged
parts of the tree)

Typical usage looks like this::

    #!/usr/bin/env python

    from scenicoverlook import MapReduceLogic, ViewableList

    mr = MapReduceLogic(reducer=lambda x, y: x + ' ' + y, initializer='')
    l = ViewableList(['the', 'quick', 'brown', 'fox'])
    print l.map_reduce(mr)                              # 'the quick brown fox'
    print (l[:2] + ['stealthy'] + l[2:]).map_reduce(mr) # 'the quick stealthy brown fox'


See the pydocs for more examples.
