# vim: fileencoding=utf-8

import functools


_inf = float('inf')


class HyperPoint(tuple):

    def __new__(cls, *points):
        return tuple.__new__(cls, (float(i) for i in points))

    @classmethod
    def origin(cls, dimensions):
        return cls(*(0 for x in xrange(dimensions)))


def dist(p1, p2, type=None):
    assert len(p1) == len(p2)
    if type == 'm':
        return sum(abs(a-b) for a, b in zip(p1, p2))
    else:
        return sum((a-b)**2 for a, b in zip(p1, p2)) ** 0.5


class Point(HyperPoint):

    def __new__(cls, x, y):
        return HyperPoint.__new__(cls, x, y)

    origin = functools.partial(HyperPoint.origin, 2)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]


class Center(Point):

    def __new__(cls, point, bounds):
        self = Point.__new__(cls, point.x, point.y)
        self.bounds = bounds
        return self

    def extend(self, point):
        def ex(selp, selb):
            if selp(point) < selb(self.bounds)[0]:
                return selp(point), selb(self.bounds)[1]
            if selp(point) > selb(self.bounds)[1]:
                return selb(self.bounds)[0], selp(point)
            return selb(self.bounds)
        xmin, xmax = ex(*(lambda x: x[0],)*2)
        ymin, ymax = ex(*(lambda x: x[1],)*2)
        return Center(Point((xmin+xmax)/2, (ymin+ymax)/2),
                ((xmin, xmax), (ymin, ymax)))


def center(points):
    xmin, ymin, xmax, ymax = _inf, _inf, -1*_inf, -1*_inf
    for point in points:
        if point.x < xmin: xmin = point.x
        if point.x > xmax: xmax = point.x
        if point.y < ymin: ymin = point.y
        if point.y > ymax: ymax = point.y
    return Center(Point((xmin+xmax)/2, (ymin+ymax)/2),
            ((xmin, xmax), (ymin, ymax)))


