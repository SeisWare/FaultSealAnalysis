import numpy as np

from constants import log
from constants import tol_path as tol

from scipy.optimize import leastsq

import util

def fit_nsphere(points, prior=None):
    """
    Fit an n-sphere to a set of points using least squares.
    Parameters
    ------------
    points : (n, d) float
      Points in space
    prior : (d,) float
      Best guess for center of nsphere
    Returns
    ---------
    center : (d,) float
      Location of center
    radius : float
      Mean radius across circle
    error : float
      Peak to peak value of deviation from mean radius
    """
    # make sure points are numpy array
    points = np.asanyarray(points, dtype=np.float64)
    # create ones so we can dot instead of using slower sum
    ones = np.ones(points.shape[1])

    def residuals(center):
        # do the axis sum with a dot
        # this gets called a LOT so worth optimizing
        radii_sq = np.dot((points - center) ** 2, ones)
        # residuals are difference between mean
        # use our sum mean vs .mean() as it is slightly faster
        return radii_sq - (radii_sq.sum() / len(radii_sq))

    if prior is None:
        guess = points.mean(axis=0)
    else:
        guess = np.asanyarray(prior)

    center_result, return_code = leastsq(residuals,
                                         guess,
                                         xtol=1e-8)

    if not (return_code in [1, 2, 3, 4]):
        raise ValueError('Least square fit failed!')

    radii = util.row_norm(points - center_result)
    radius = radii.mean()
    error = radii.ptp()
    return center_result, radius, error


def fit_circle_check(points,
                     scale,
                     prior=None,
                     final=False,
                     verbose=False):
    """
    Fit a circle, and reject the fit if:
    * the radius is larger than tol.radius_min*scale or tol.radius_max*scale
    * any segment spans more than tol.seg_angle
    * any segment is longer than tol.seg_frac*scale
    * the fit deviates by more than tol.radius_frac*radius
    * the segments on the ends deviate from tangent by more than tol.tangent
    Parameters
    ---------
    points :  (n, d)
      List of points which represent a path
    prior :  (center, radius) tuple
      Best guess or None if unknown
    scale : float
      What is the overall scale of the set of points
    verbose : bool
     Output log.debug messages for the reasons
     for fit rejection only suggested for manual debugging
    Returns
    -----------
    if fit is acceptable:
        (center, radius) tuple
    else:
        None
    """
    # an arc needs at least three points
    if len(points) < 3:
        return None
    # make sure our points are a numpy array
    points = np.asanyarray(points, dtype=np.float64)

    # do a least squares fit on the points
    C, R, r_deviation = fit_nsphere(points, prior=prior)

    # check to make sure radius is between min and max allowed
    if not tol.radius_min < (R / scale) < tol.radius_max:
        if verbose:
            log.debug('circle fit error: R %f', R / scale)
        return None

    # check point radius error
    r_error = r_deviation / R
    if r_error > tol.radius_frac:
        if verbose:
            log.debug('circle fit error: fit %s', str(r_error))
        return None

    vectors = np.diff(points, axis=0)
    segment = util.row_norm(vectors)

    # approximate angle in radians, segments are linear length
    # not arc length but this is close and avoids a cosine
    angle = segment / R
    if (angle > tol.seg_angle).any():
        if verbose:
            log.debug('circle fit error: angle %s', str(angle))
        return None

    if final and (angle > tol.seg_angle_min).sum() < 3:
        log.debug('final: angle %s', str(angle))
        return None

    # check segment length as a fraction of drawing scale
    scaled = segment / scale

    if (scaled > tol.seg_frac).any():
        if verbose:
            log.debug('circle fit error: segment %s', str(scaled))
        return None

    # check to make sure the line segments on the ends are actually
    # tangent with the candidate circle fit
    mid_pt = points[[0, -2]] + (vectors[[0, -1]] * .5)
    radial = util.unitize(mid_pt - C)
    ends = util.unitize(vectors[[0, -1]])
    tangent = np.abs(np.arccos(util.diagonal_dot(radial, ends)))
    tangent = np.abs(tangent - np.pi / 2).max()

    if tangent > tol.tangent:
        if verbose:
            log.debug('circle fit error: tangent %f',
                      np.degrees(tangent))
        return None

    result = {'center': C,
              'radius': R}

    return result



def medial_axis(polygon,
                resolution=None,
                clip=None):
    """
    Given a shapely polygon, find the approximate medial axis
    using a voronoi diagram of evenly spaced points on the
    boundary of the polygon.
    Parameters
    ----------
    polygon : shapely.geometry.Polygon
      The source geometry
    resolution : float
      Distance between each sample on the polygon boundary
    clip : None, or (2,) int
      Clip sample count to min of clip[0] and max of clip[1]
    Returns
    ----------
    edges : (n, 2) int
      Vertex indices representing line segments
      on the polygon's medial axis
    vertices : (m, 2) float
      Vertex positions in space
    """
    # a circle will have a single point medial axis
    if len(polygon.interiors) == 0:
        # what is the approximate scale of the polygon
        scale = np.reshape(polygon.bounds, (2, 2)).ptp(axis=0).max()
        # a (center, radius, error) tuple
        fit = fit_circle_check(
            polygon.exterior.coords, scale=scale)
        # is this polygon in fact a circle
        if fit is not None:
            # return an edge that has the center as the midpoint
            epsilon = np.clip(
                fit['radius'] / 500, 1e-5, np.inf)
            vertices = np.array(
                [fit['center'] + [0, epsilon],
                 fit['center'] - [0, epsilon]],
                dtype=np.float64)
            # return a single edge to avoid consumers needing to special case
            edges = np.array([[0, 1]], dtype=np.int64)
            return edges, vertices

    from scipy.spatial import Voronoi
    from shapely import vectorized

    if resolution is None:
        resolution = np.reshape(
            polygon.bounds, (2, 2)).ptp(axis=0).max() / 100

    # get evenly spaced points on the polygons boundaries
    samples = resample_boundaries(polygon=polygon,
                                  resolution=resolution,
                                  clip=clip)
    # stack the boundary into a (m,2) float array
    samples = stack_boundaries(samples)
    # create the voronoi diagram on 2D points
    voronoi = Voronoi(samples)
    # which voronoi vertices are contained inside the polygon
    contains = vectorized.contains(polygon, *voronoi.vertices.T)
    # ridge vertices of -1 are outside, make sure they are False
    contains = np.append(contains, False)
    # make sure ridge vertices is numpy array
    ridge = np.asanyarray(voronoi.ridge_vertices, dtype=np.int64)
    # only take ridges where every vertex is contained
    edges = ridge[contains[ridge].all(axis=1)]

    # now we need to remove uncontained vertices
    contained = np.unique(edges)
    mask = np.zeros(len(voronoi.vertices), dtype=np.int64)
    mask[contained] = np.arange(len(contained))

    # mask voronoi vertices
    vertices = voronoi.vertices[contained]
    # re-index edges
    edges_final = mask[edges]

    if tol.strict:
        # make sure we didn't screw up indexes
        assert (vertices[edges_final] -
                voronoi.vertices[edges]).ptp() < 1e-5

    return edges_final, vertices


def resample_boundaries(polygon, resolution, clip=None):
    """
    Return a version of a polygon with boundaries resampled
    to a specified resolution.
    Parameters
    -------------
    polygon : shapely.geometry.Polygon
      Source geometry
    resolution : float
      Desired distance between points on boundary
    clip : (2,) int
      Upper and lower bounds to clip
      number of samples to avoid exploding count
    Returns
    ------------
    kwargs : dict
     Keyword args for a Polygon constructor `Polygon(**kwargs)`
    """
    def resample_boundary(boundary):
        # add a polygon.exterior or polygon.interior to
        # the deque after resampling based on our resolution
        count = boundary.length / resolution
        count = int(np.clip(count, *clip))
        return resample_path(boundary.coords, count=count)
    if clip is None:
        clip = [8, 200]
    # create a sequence of [(n,2)] points
    kwargs = {'shell': resample_boundary(polygon.exterior),
              'holes': []}
    for interior in polygon.interiors:
        kwargs['holes'].append(resample_boundary(interior))

    return kwargs



def stack_boundaries(boundaries):
    """
    Stack the boundaries of a polygon into a single
    (n, 2) list of vertices.
    Parameters
    ------------
    boundaries : dict
      With keys 'shell', 'holes'
    Returns
    ------------
    stacked : (n, 2) float
      Stacked vertices
    """
    if len(boundaries['holes']) == 0:
        return boundaries['shell']
    result = np.vstack((boundaries['shell'],
                        np.vstack(boundaries['holes'])))
    return result



def resample_path(points,
                  count=None,
                  step=None,
                  step_round=True):
    """
    Given a path along (n,d) points, resample them such that the
    distance traversed along the path is constant in between each
    of the resampled points. Note that this can produce clipping at
    corners, as the original vertices are NOT guaranteed to be in the
    new, resampled path.
    ONLY ONE of count or step can be specified
    Result can be uniformly distributed (np.linspace) by specifying count
    Result can have a specific distance (np.arange) by specifying step
    Parameters
    ----------
    points:   (n, d) float
        Points in space
    count : int,
        Number of points to sample evenly (aka np.linspace)
    step : float
        Distance each step should take along the path (aka np.arange)
    Returns
    ----------
    resampled : (j,d) float
        Points on the path
    """

    points = np.array(points, dtype=np.float64)
    # generate samples along the perimeter from kwarg count or step
    if (count is not None) and (step is not None):
        raise ValueError('Only step OR count can be specified')
    if (count is None) and (step is None):
        raise ValueError('Either step or count must be specified')

    sampler = PathSample(points)
    if step is not None and step_round:
        if step >= sampler.length:
            return points[[0, -1]]

        count = int(np.ceil(sampler.length / step))

    if count is not None:
        samples = np.linspace(0, sampler.length, count)
    elif step is not None:
        samples = np.arange(0, sampler.length, step)

    resampled = sampler.sample(samples)

    check = util.row_norm(points[[0, -1]] - resampled[[0, -1]])
    assert check[0] < tol.merge
    if count is not None:
        assert check[1] < tol.merge

    return resampled


class PathSample:

    def __init__(self, points):
        # make sure input array is numpy
        self._points = np.array(points)
        # find the direction of each segment
        self._vectors = np.diff(self._points, axis=0)
        # find the length of each segment
        self._norms = util.row_norm(self._vectors)
        # unit vectors for each segment
        nonzero = self._norms > tol.zero
        self._unit_vec = self._vectors.copy()
        self._unit_vec[nonzero] /= self._norms[nonzero].reshape((-1, 1))
        # total distance in the path
        self.length = self._norms.sum()
        # cumulative sum of section length
        # note that this is sorted
        self._cum_norm = np.cumsum(self._norms)

    def sample(self, distances):
        # return the indices in cum_norm that each sample would
        # need to be inserted at to maintain the sorted property
        positions = np.searchsorted(self._cum_norm, distances)
        positions = np.clip(positions, 0, len(self._unit_vec) - 1)
        offsets = np.append(0, self._cum_norm)[positions]
        # the distance past the reference vertex we need to travel
        projection = distances - offsets
        # find out which dirction we need to project
        direction = self._unit_vec[positions]
        # find out which vertex we're offset from
        origin = self._points[positions]
        # just the parametric equation for a line
        resampled = origin + (direction * projection.reshape((-1, 1)))

        return resampled

    def truncate(self, distance):
        """
        Return a truncated version of the path.
        Only one vertex (at the endpoint) will be added.
        """
        position = np.searchsorted(self._cum_norm, distance)
        offset = distance - self._cum_norm[position - 1]

        if offset < constants.tol_path.merge:
            truncated = self._points[:position + 1]
        else:
            vector = util.unitize(np.diff(
                self._points[np.arange(2) + position],
                axis=0).reshape(-1))
            vector *= offset
            endpoint = self._points[position] + vector
            truncated = np.vstack((self._points[:position + 1],
                                   endpoint))
        assert (util.row_norm(np.diff(
            truncated, axis=0)).sum() -
            distance) < tol.merge

        return truncated
