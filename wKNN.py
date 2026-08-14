import numpy as np
from collections import Counter


def weighted_euclidean_distance(p,q) :

  pass

class KNearestNeighbors:

  def __init__(self, k =3):
    self.k = k
    self.points = None

     
  def fit(self):
    self.points = points
    return self

  def predict(self , new_points) :

    distances = [[weighted_euclidean_distance(point , new_points), category] for category in self.points for point in self.points[category]]

    labels = [category[1] for category in sorted(distances)[:self.k]]

    result = Counter(labels).most_common(1)[0][0]

    return result

  