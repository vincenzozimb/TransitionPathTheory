import numpy as np

# Create your long vector (replace this with your actual vector)
long_vector = np.array([0, 0, 1, 2, 0, 0, 3, 0, 4, 5, 0, 0, 0, 6, 7, 8, 0])

# Find the indices of non-zero elements
non_zero_indices = np.nonzero(long_vector)[0]

# # Split the non-zero indices into segments
# segments = np.split(non_zero_indices, np.where(np.diff(non_zero_indices) != 1)[0] + 1)

# # Extract non-zero pieces into separate vectors
# non_zero_vectors = [long_vector[segment] for segment in segments]

# print(non_zero_vectors)

