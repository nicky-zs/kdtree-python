# vim: fileencoding=utf-8

import functools


_inf = float('inf')


class Interval(object):

    def __init__(self, lower, upper):
        self.set_to(lower, upper)

    def __repr__(self):
        return '[%g, %g]' % (self.lower, self.upper)

    def contains(self, p):
        return self.lower <= p <= self.upper

    def set_to(self, lower, upper):
        self.lower = float(lower)
        self.upper = float(upper)

    def enlarge_to(self, p):
        if self.lower > p:
            self.lower = p
        elif self.upper < p:
            self.upper = p

    def clone(self):
        return Interval(self.lower, self.upper)


def _check_dim(func):
    @functools.wraps(func)
    def wrapper(self, hpoint, *args, **kws):
        assert len(self) == len(hpoint)
        return func(self, hpoint, *args, **kws)
    return wrapper


class HyperRect(tuple):

    def __new__(cls, *intervals):
        return tuple.__new__(cls, intervals)

    @classmethod
    def space(cls, dimensions):
        return cls(*(Interval(-1*_inf, _inf) for x in xrange(dimensions)))

    @classmethod
    def point(cls, point):
        return cls(*(Interval(x, x) for x in point))

    def __repr__(self):
        return '(%s)' % ','.join(repr(x) for x in self)

    def clone(self):
        return HyperRect(*(i.clone() for i in self))

    @_check_dim
    def contains(self, hpoint):
        return all(interval.contains(p) for interval, p in zip(self, hpoint))

    @_check_dim
    def min_dist(self, hpoint):
        return sum((p-itv.lower if p<itv.lower else p-itv.upper)**2
                for itv, p in zip(self, hpoint) if not itv.contains(p)) ** 0.5

    @_check_dim
    def enlarge_to(self, hpoint):
        for interval, p in zip(self, hpoint):
            interval.enlarge_to(p)

    @_check_dim
    def get_upper(self, hpoint, _disc):
        interval, p = self[_disc], hpoint[_disc]
        if interval.upper < p:
            return None
        rect = self.clone()
        interval = rect[_disc]
        if interval.lower < p:
            interval.lower = p
        return rect

    @_check_dim
    def get_lower(self, hpoint, _disc):
        interval, p = self[_disc], hpoint[_disc]
        if interval.lower > p:
            return None
        rect = self.clone()
        interval = rect[_disc]
        if interval.upper > p:
            interval.upper = p
        return rect


class Rect(HyperRect):

    def __new__(cls, x_interval, y_interval):
        return HyperRect(x_interval, y_interval)

    space = functools.partial(HyperRect.space, 2)


