import numpy as np
from evaluations.clustering import clustering_metric

labels = np.array(  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
cluster = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
topic = np.array(   [0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 2, 3, 3, 4])

print(clustering_metric(labels, cluster))
print(clustering_metric(labels, topic))