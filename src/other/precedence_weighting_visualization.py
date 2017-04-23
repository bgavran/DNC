import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

"""
Visualization of precedence weighting
"""

memory_size = 5
time_steps = 15


def compute_precedence(precedence_old, write_weighting):
    return precedence_old * (1 - np.sum(write_weighting)) + write_weighting


def compute_link_matrix(link_matrix_old, precedence_old, write_weighting):
    write_weighting = write_weighting[:, np.newaxis]
    # The commented failure of the commented out assertion below doesn't really mean anything useful, since it will
    # sometimes fail on diagional entries of the link matrix, which are self links and therefore excluded
    # assert np.all(1 - write_weighting - write_weighting.T >= 0)
    assert np.all(np.einsum("ix, j->ij", write_weighting, precedence_old) >= 0)
    return ((1 - write_weighting - write_weighting.T) * link_matrix_old +
            np.einsum("ix,j->ij", write_weighting, precedence_old)) * (1 - np.eye(memory_size, memory_size))


write_pattern = np.zeros((memory_size, time_steps))
for t in range(time_steps):
    loc = np.random.randint(0, memory_size)
    write_pattern[loc, t] = 1

noise_multiplier = 0.1
write_gate = 0.6
write_weighting_unnormalized = np.random.rand(memory_size, time_steps) * noise_multiplier + write_pattern
write_weighting = write_gate * write_weighting_unnormalized / np.sum(write_weighting_unnormalized, axis=0)

precedence = np.zeros((memory_size, time_steps))
link_matrix = np.zeros((time_steps, memory_size, memory_size))

for i in range(1, time_steps):
    precedence[:, i] = compute_precedence(precedence[:, i - 1], write_weighting[:, i])
    link_matrix[i] = compute_link_matrix(link_matrix[i - 1], precedence[:, i - 1], write_weighting[:, i])

forward_weights = link_matrix[-1] @ np.array([[0, 1, 0, 0, 1]]).T
l = [write_weighting, link_matrix[-1], forward_weights]
f, axes = plt.subplots(len(l))
kwargs = {"annot": False, "fmt": ".2f"}
for i, data in enumerate(l):
    if len(l) == 1:
        ax = axes
    else:
        ax = axes[i]
    sns.heatmap(data, ax=ax, **kwargs)

plt.show()
