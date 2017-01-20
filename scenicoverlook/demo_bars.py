import collections, random, sys, time
from viewablelist import viewablelist
from mapreducelogic import MapReduceLogic

num_bars = int(sys.argv[1]) if len(sys.argv) == 2 else 25
total_width = 500
Bar = collections.namedtuple('bar', ['x', 'w', 'fillchar'])
r = random.Random()
def random_bar():
    return Bar(
        x=r.randint(0, 79),
        w=r.randint(4, 10),
        fillchar=r.choice("^#$%&.':~_=@|-+*/\()[]<>{}") # some fun characters
    )

bars = viewablelist([random_bar() for _ in range(num_bars)])

def bar_to_line(b):
    # turn a bar into a line of text (mostly spaces)
    return [b.fillchar if b.x <= i < b.x + b.w else ' ' for i in range(80)]

CTR = 0
def overlay_lines(base, overlay):
    # overlay one line of text atop another (spaces are transparent)
    global CTR
    CTR += 1
    return [base[i] if overlay[i] == ' ' else overlay[i] for i in range(80)]

mr = MapReduceLogic(mapper = bar_to_line, reducer = overlay_lines)

while bars:
    CTR = 0
    line = bars.map_reduce(mr)
    print ''.join(line), '  (num bars: %05d reduces called: %03d)' %(len(bars),CTR)
    time.sleep(0.011)
    last = bars[-1]
    bars = bars[:-1]
    if last.x < 80:
        moved_bar = Bar(x = last.x + 1, w = last.w, fillchar=last.fillchar)
        bars = bars + [moved_bar]
