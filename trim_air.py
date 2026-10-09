import numpy as np

# Load arrays
perms = np.load('perm_list_multi_2D_pep_rs.npy')

perms_trimmed = perms[:, :190, :]

np.save('perm_list_multi_2D_pep_rs_trim.npy',perms_trimmed)
