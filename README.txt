===============
Scenic Overlook
===============

The Scenic Overlook library contains datastructures for incremental
map-reduces.

You might want to read
[my blog post about the general problem](https://hackernoon.com/computed-state-the-model-view-problem-9cbe8cf8486f).

These datastructures are implemented as trees, and store at each node,
intermediate values of the reduce. This means that when you slice or combine
structures, the new output of the maps/reduces can be efficiently computed.
(by reusing old outputs from unchanged parts of the tree)

Typical usage looks like this::

    #!/usr/bin/env python

    from scenicoverlook import viewablelist

    space_concat = lambda x, y: x + ' ' + y
    l = viewablelist(['the', 'quick', 'brown', 'fox'])
    print l.reduce(space_concat)
    
    # This yields 'the quick stealthy brown fox', reusing cached intermediate
    # substrings from the earlier call like 'the quick' and 'brown fox':
    
    print (l[:2] + ['stealthy'] + l[2:]).reduce(space_concat)


See the pydocs for more examples:

https://github.com/pschanely/ScenicOverlook/blob/master/scenicoverlook/__init__.py
