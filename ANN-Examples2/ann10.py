# Write a python program to illustrate ART neural network. 

import numpy as np

def compute_similarity(c, x):
    intersection = np.logical_and(c, x)
    similarity = np.sum(intersection) / np.sum(x)
    return intersection, similarity

def art_network(x, vigilance):
    cluster = []
    for sample in x:
        assigned = False
        for i, c in enumerate(cluster):
            intersection, similarity = compute_similarity(c, sample)
            if similarity >= vigilance:
                cluster[i] = np.logical_and(c, sample)
                assigned = True
                break
        if not assigned:
            cluster.append(sample.copy())
    return cluster

x = np.array([
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1]
])

vigilance = 0.6
final_clusters = art_network(x, vigilance)

print("\nFinal Learned Categories:")
for i, c in enumerate(final_clusters):
    print(f"Category {i + 1}: {c}")
