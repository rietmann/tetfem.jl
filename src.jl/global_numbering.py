#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Routines to get global numbering

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import numpy as np


def get_global_lexi(points, tolerance_decimals=8):
    """
    get global numbering scheme based on lexicographic sorting

    Note that this method does not preserve previously existing sorting for
    points that are readily unique.

    :param points: points in ndim dimensional space stored in array with shape
        (ndim, npoints)
    :type points: numpy array
    :param tolerance_decimals: accuracy to assume for two points to be equal
        after renormalization of coordinates
    :type tolerance_decimals: integer

    :returns: tupel of the global indices as numpy integer array and shape
        (npoint,) and the number of global points
    """

    # do not work inplace here:
    points = points.copy()

    # initialization:
    ndim, npoints = points.shape
    segstart = np.array([0])
    segend = np.array([npoints])
    tolerance = 10. ** (- tolerance_decimals)

    # find maximum spread in all dimensions
    maxmax = np.max([np.ptp(points[dim, :]) for dim in np.arange(ndim)])

    # compute absolute tolerances
    tolerance /= maxmax

    # array to keep track of the reshuffling
    loc = np.arange(npoints)

    # sort lexicographically in all dimensions, where in higher iterations only
    # those points that are the same in lower dimensions (within tolerance) are
    # sorted. This is not only faster, but also ensures the functionality of
    # the floating point tolerance.
    for dim in np.arange(ndim):
        # sort in each segment
        for s, e in zip(segstart, segend):
            sort_id = points[dim, s:e].argsort()
            points[:, s:e] = points[:, s:e][:, sort_id]
            loc[s:e] = loc[s:e][sort_id]

        # update segments of same points
        segments = np.zeros(npoints + 1, dtype='bool')
        segments[1:-1] = np.diff(points[dim, :]) < tolerance
        diff = np.diff(segments.astype('int'))
        segstart = np.where(diff == 1)[0]
        segend = np.where(diff == -1)[0] + 1

    # compute distance between neighbours:
    dist_square = ((points[:, 1:] - points[:, :-1]) ** 2).sum(axis=0)

    # generate global index
    global_index = np.zeros(npoints, dtype='int')
    global_index[1:] = (dist_square > tolerance ** 2).cumsum()

    # total number of distinct points
    nglob = global_index[-1] + 1

    # resort index to the original sorting of points
    global_index = global_index[loc]

    return global_index, nglob


class LookupTreeNode(object):
    def __init__(self, point, index):
        self.point = point
        self.index = index
        self.child1 = None
        self.child2 = None

    def is_leaf(self):
        return (self.child1 is None and self.child2 is None)

    def dist(self, point):
        return ((self.point - point) ** 2).sum() ** 0.5

    def is_same(self, point, tolerance=1e-8):
        return self.dist(point) < tolerance

    def find(self, point, tolerance=1e-8):
        """
        Find a point in the lookup tree with some floating point tolerance. If
        found, the index is returned, otherwise None.

        :param points: point in ndim dimensional space stored in array with
            shape (ndim, )
        :type points: numpy array
        """
        if self.is_same(point, tolerance):
            return self.index

        elif self.is_leaf():
            return None

        # if only one child is initialized, it is always child 1
        elif self.child2 is None:
            return self.child1.find(point, tolerance)

        # both childs are initialized
        else:
            # compute distance two both childs
            dist1 = self.child1.dist(point)
            dist2 = self.child2.dist(point)

            # if within tolerance of same distance to both points (i.e. on the
            # cutting plane), search on both sides:
            if abs(dist1 - dist2) < tolerance:
                return (self.child1.find(point, tolerance) or
                        self.child2.find(point, tolerance))

            # otherwise, search on closer side
            elif (dist1 < dist2):
                return self.child1.find(point, tolerance)
            else:
                return self.child2.find(point, tolerance)

    def find_or_insert(self, point, index_next, tolerance=1e-8):
        """
        Find a point in the lookup tree with some floating point tolerance. If
        found, the index is returned, otherwise the point is inserted in the
        tree.

        :param points: point in ndim dimensional space stored in array with
            shape (ndim, )
        :type points: numpy array

        :returns: tupel of the global index of the point and the next index to
            be used when inserting another point
        """
        if self.is_same(point, tolerance):
            return self.index, index_next

        elif self.is_leaf():
            self.child1 = LookupTreeNode(point, index_next)
            index_now = index_next
            index_next += 1
            return index_now, index_next

        # if only one child is initialized, it is always child 1
        elif self.child2 is None:
            if self.child1.is_same(point, tolerance):
                return self.child1.index, index_next

            else:
                self.child2 = LookupTreeNode(point, index_next)
                index_now = index_next
                index_next += 1
                return index_now, index_next

        # both childs are initialized
        else:
            # compute distance two both childs
            dist1 = self.child1.dist(point)
            dist2 = self.child2.dist(point)

            # if within tolerance of same distance to both points (i.e. on
            # the cutting plane), search on both sides:
            if abs(dist1 - dist2) < tolerance:
                index = (self.child1.find(point, tolerance) or
                         self.child2.find(point, tolerance))

                # if we found an equivalent point, return it:
                if index is not None:
                    return index, index_next

            # otherwise, continue inserting on closer side:
            if (dist1 < dist2):
                return self.child1.find_or_insert(point, index_next, tolerance)
            else:
                return self.child2.find_or_insert(point, index_next, tolerance)

def get_global_tree(points, tolerance_decimals=8):
    """
    get global numbering scheme based on a binary lookup tree

    Note that this method preserves previously existing sorting for points that
    are readily unique and might hence be beneficial for cache effects.

    :param points: points in ndim dimensional space stored in array with shape
        (ndim, npoints)
    :type points: numpy array
    :param tolerance_decimals: accuracy to assume for two points to be equal
        after renormalization of coordinates
    :type tolerance_decimals: integer

    :returns: tupel of the global indices as numpy integer array and shape
        (npoint,) and the number of global points
    """

    # do not work inplace here:
    points = points.copy()

    # initialization:
    ndim, npoints = points.shape
    tolerance = 10. ** (- tolerance_decimals)
    global_index = np.zeros(npoints, dtype='int')

    # find maximum spread in all dimensions
    maxmax = np.max([np.ptp(points[dim, :]) for dim in np.arange(ndim)])

    # compute absolute tolerances
    tolerance /= maxmax

    root = LookupTreeNode(points[:, 0], 0)
    index_next = 1

    # find global index by attempting to insert into the tree
    for i, point in enumerate(points.T[1:]):
        global_index[i + 1], index_next = root.find_or_insert(point, index_next, tolerance)

    # index next now is the number of distinct points
    return global_index, index_next
