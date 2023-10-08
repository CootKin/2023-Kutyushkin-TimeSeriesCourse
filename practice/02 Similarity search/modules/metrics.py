import numpy as np
import math

def ED_distance(ts1, ts2):
    ed_dist = 0

    for item1, item2 in zip(ts1, ts2):
      ed_dist += (item1 - item2)**2
    
    return math.sqrt(ed_dist)

def arithmeticMeanDev(ts):
  return np.sum(ts)/len(ts)


def standartDev(ts):
  s = 0
  mean_square = arithmeticMeanDev(ts)**2
  for item in ts:
    s += item**2 - mean_square
  
  return math.sqrt(s / len(ts))


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """
    norm_ed_dist = 0

    n = len(ts1)
    mu1 = arithmeticMeanDev(ts1)
    mu2 = arithmeticMeanDev(ts2)
    sigma1 = standartDev(ts1)
    sigma2 = standartDev(ts2)
    
    ed = (np.dot(ts1, ts2) - n*mu1*mu2)/(n*sigma1*sigma2)

    norm_ed_dist = math.sqrt(abs(2*n*(1 - ed)))

    return norm_ed_dist


def DTW_distance(ts1, ts2, r=None):
  
    n = len(ts1)
    m = len(ts2)

    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[:, :] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(max(1, i-int(np.floor(m*r))), min(m, i+int(np.floor(m*r))) + 1):
            cost = np.square(ts1[i-1] - ts2[j-1])
            dtw_matrix[i, j] = cost + \
                min(dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1])

    dtw_dist = dtw_matrix[n, m]

    return dtw_dist

    