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
    """
    r : float
        Warping window size.
    """
    m = len(ts1)

    D = np.zeros(shape=(m, m))

    for i in range(0, m):
      for j in range(0, m):
        if (i == 0 and j == 0):
          D[i][j] = (ts1[i] - ts2[j])**2
        elif (i == 0):
          D[i][j] = (ts1[i] - ts2[j])**2 + D[i][j - 1]
        elif (j == 0):
          D[i][j] = (ts1[i] - ts2[j])**2 + D[i - 1][j]
        else:
          D[i][j] = (ts1[i] - ts2[j])**2 + np.min([D[i-1][j],
                                    D[i][j-1],
                                    D[i-1][j-1]])

    return float(D[m-1][m-1])

    