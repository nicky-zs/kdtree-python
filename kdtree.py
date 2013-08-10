# vim: fileencoding=utf-8

import random

from rect import HyperRect


# Multidimensional Binary Search Trees Used for Associative Searching, Jon Louis Bentley


def kdtree(dimension):
    return KDTree(dimension)

def static_kdtree(points):
    return StaticKDTree(points)

def euclidean_dist(p1, p2):
    assert len(p1) == len(p2)
    return sum((a-b)**2 for a, b in zip(p1, p2)) ** 0.5


LOSON, HISON = 0, 1

class KDNode(object):

    def __init__(self, point=None):
        self.point = point
        self.parent = None
        self.children = [None, None]

    @property
    def loson(self):
        return self.children[LOSON]

    @loson.setter
    def loson(self, value):
        self.children[LOSON] = value

    @property
    def hison(self):
        return self.children[HISON]

    @hison.setter
    def hison(self, value):
        self.children[HISON] = value

    def __len__(self):
        return len(self.point)

    def  __repr__(self):
        return '<KDNode at %r, loson:%r, hison:%r]>' % (self.point,
                self.loson and self.loson.point or None, self.hison and self.hison.point or None)


class KDTree(object):

    def __init__(self, dimensions):
        self.dim = dimensions
        self.root = None
        self._rect = None
        self._size = 0

    def __len__(self):
        return self._size

    def __repr__(self):
        return '<KDTree with root: %r>' % self.root

    def _successor(self, node, point, _disc):
        for i in range(_disc, self.dim) + range(0, _disc):
            if point[i] < node.point[i]:
                return LOSON
            if point[i] > node.point[i]:
                return HISON
        raise Exception('the point is the same with node.point')

    def _minimum(self, node, disc, _disc):
        _node, _d = node, _disc
        _disc_next = (_disc + 1) % self.dim
        if _disc == disc:
            child = node.loson
            if child:
                _node, _d = self._minimum(child, disc, _disc_next)
        else:
            if any(node.children):
                _node, _d = min([self._minimum(child, disc, _disc_next) for child in node.children if child] + [(node, _disc)],
                    key=lambda node_disc: node_disc[0].point[disc])
        return _node, _d

    def _maximum(self, node, disc, _disc):
        _node, _d = node, _disc
        _disc_next = (_disc + 1) % self.dim
        if _disc == disc:
            child = node.hison
            if child:
                _node, _d = self._maximum(child, disc, _disc_next)
        else:
            if any(node.children):
                _node, _d = max([self._maximum(child, disc, _disc_next) for child in node.children if child] + [(node, _disc)],
                    key=lambda node_disc: node_disc[0].point[disc])
        return _node, _d

    def insert(self, point):

        ''' insert a point into kdtree '''

        new_node = KDNode(point)
        node, _disc = self.root, 0
        if not node:
            self.root = new_node
            self._rect = HyperRect.point(point)
        else:
            while True:
                if node.point == point:
                    return node
                son = self._successor(node, point, _disc)
                child = node.children[son]
                if not child:
                    break
                node = child
                _disc = (_disc + 1) % self.dim
            node.children[son] = new_node
            new_node.parent = node
            self._rect.enlarge_to(point)
        self._size += 1
        return None

    def _delete(self, node, _disc):
        if not node.loson and not node.hison:
            if node.parent:
                son = LOSON if node.parent.loson is node else HISON
                node.parent.children[son] = None
            return None
        disc = _disc
        _disc = (_disc + 1) % self.dim
        q, d = self._minimum(node.hison, disc, _disc) if node.hison else self._maximum(node.loson, disc, _disc)
        node.point = q.point
        qfather = q.parent
        qson = LOSON if qfather.loson is q else HISON
        qfather.children[qson] = self._delete(q, d)
        return node

    def delete(self, point):

        ''' delete a point from the kdtree '''

        node, _disc = self.root, 0
        while node:
            if node.point == point:
                break
            node = node.children[self._successor(node, point, _disc)]
            _disc = (_disc + 1) % self.dim
        if node:
            if node is self.root:
                self.root = self._delete(node, _disc)
            else:
                self._delete(node, _disc)
            self._size -= 1
            if self._size == 1:
                self._rect = HyperRect.point(point)
            elif self._size == 0:
                self._rect = None
        return node

    def _nn(self, node, point, _rect, dist, best, _disc):
        if not node or _rect.min_dist(point) > dist:
            return float('inf'), None
        _disc_next = (_disc + 1) % self.dim
        dist, best = min((euclidean_dist(node.point, point), node.point), (dist, best))
        lower = _rect.get_lower(node.point, _disc)
        upper = _rect.get_upper(node.point, _disc)
        if point[_disc] < node.point[_disc]:
            dist, best = min(self._nn(node.loson, point, lower, dist, best, _disc_next), (dist, best))
            dist, best = min(self._nn(node.hison, point, upper, dist, best, _disc_next), (dist, best))
        else:
            dist, best = min(self._nn(node.hison, point, upper, dist, best, _disc_next), (dist, best))
            dist, best = min(self._nn(node.loson, point, lower, dist, best, _disc_next), (dist, best))
        return dist, best

    def nn(self, point):

        ''' find the nearest neighbour of the point '''

        return self._nn(self.root, point, self._rect, float('inf'), None, 0)[1]

    def nn_with_dist(self, point):

        ''' find the nearest neighbour of the point, returned with the dist '''

        return self._nn(self.root, point, self._rect, float('inf'), None, 0)

    def _knn(self, node, point, _rect, dist, best, _disc):
        if not node or _rect.min_dist(point) > dist:
            return
        _disc_next = (_disc + 1) % self.dim
        d = euclidean_dist(node.point, point)
        if d <= dist:
            best.append((d, node.point))
        lower = _rect.get_lower(node.point, _disc)
        upper = _rect.get_upper(node.point, _disc)
        self._knn(node.loson, point, lower, dist, best, _disc_next)
        self._knn(node.hison, point, upper, dist, best, _disc_next)

    def knn(self, point, dist):

        ''' find the k nearest neighbours(ordered) of the point '''

        best = []
        self._knn(self.root, point, self._rect, dist, best, 0)
        return [x[1] for x in sorted(best)]

    def knn_with_dist(self, point, dist):

        ''' find the k nearest neighbours(ordered) of the point,
        returned with dists '''

        best = []
        self._knn(self.root, point, self._rect, dist, best, 0)
        return sorted(best)


class StaticKDTree(KDTree):

    def __init__(self, points):
        if not points:
            raise Exception('points must NOT be empty')
        points = [p for p in points]
        random.shuffle(points)

        super(StaticKDTree, self).__init__(len(points[0]))
        self._size = len(points)
        self._rect = HyperRect.point(points[0])
        for p in points:
            self._rect.enlarge_to(p)
        self.root = self._build(points, 0, 0, len(points)-1)

    def _build(self, points, _disc, p, r):
        if p > r:
            return None

        m = self._median(points, _disc, p, r)
        n = KDNode(points[m])
        _disc = (_disc + 1) % self.dim
        l = self._build(points, _disc, p, m-1)
        h = self._build(points, _disc, m+1, r)
        if l:
            n.loson = l
            l.parent = n
        if h:
            n.hison = h
            h.parent = n
        return n

    def _median(self, points, _disc, p, r):
        return self._select(points, _disc, p, r, (r-p)/2+1)

    def _select(self, points, _disc, p, r, i):
        if p == r:
            return p
        q = self._partition(points, _disc, p, r)
        k = q - p + 1
        if i == k:
            return q
        elif i < k:
            return self._select(points, _disc, p, q-1, i)
        else:
            return self._select(points, _disc, q+1, r, i-k)

    def _partition(self, points, _disc, p, r):
        x = points[r]
        i = p - 1
        for j in xrange(p, r):
            if points[j][_disc] <= x[_disc]:
                i += 1
                temp = points[i]
                points[i] = points[j]
                points[j] = temp
        i += 1
        temp = points[i]
        points[i] = points[r]
        points[r] = temp
        return i

    def _not_supported(*a, **kw):
        raise Exception('this method is not supported')

    insert = delete = _not_supported


def insert_delete_test():
    N = 1000000
    r = functools.partial(random.randint, 1, N)

    count = 0
    while True:
        if count % 10 == 0:
            pass
            #print 'test count: %d' % count

        tree = KDTree(2)
        points, ps = [], set()
        for x in xrange(10000):
            i = r()
            if i not in ps:
                points.append(Point(i, i))
                ps.add(i)

        for p in points:
            tree.insert(p)

        if len(tree) != len(points):
            print 'ERROR: inserting error'
            print 'points is %r' % points
            break

        nn = tree.nn_with_dist(Point(N/2, N/2))
        print nn
        print tree.knn(Point(N/2, N/2), nn[0]*2)

        #random.shuffle(points)
        for p in points:
            tree.delete(p)

        if len(tree) or tree.root:
            print 'ERROR: deleting error'
            print 'points is %r' % points
            break

        count += 1


def nn_test():
    tree = KDTree(2)
    tree.insert(Point(3, 3))
    tree.insert(Point(1, 1))
    tree.insert(Point(2, 2))

    print tree.nn(Point(0.5, 0.5))


def knn_test():
    tree = KDTree(2)
    tree.insert(Point(3, 3))
    tree.insert(Point(1, 1))
    tree.insert(Point(2, 2))

    print tree.knn(Point(0, 0), 4.5)


def static_test():
    points = []
    points.append(Point(3, 3))
    points.append(Point(1, 1))
    points.append(Point(2, 2))
    tree = static_kdtree(points)
    print tree


if __name__ == '__main__':
    import random, functools
    from point import Point

    static_test()
    #insert_delete_test()
    #knn_test()


