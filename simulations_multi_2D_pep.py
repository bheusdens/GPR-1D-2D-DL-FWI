import sys
import h5py
import numpy as np
import os
from pathlib import Path
from scipy.stats import mode

data_folder = "2D_Data_Gen_Pep_Multi"

def downsample_mode(arr, block_size=7):
    # arr: 3D numpy array (x, y, z)
    sx, sy, sz = arr.shape
    # Ensure dimensions are divisible by block_size
    nx, ny, nz = sx // block_size, sy // block_size, sz // block_size
    arr = arr[:nx*block_size, :ny*block_size, :nz*block_size]
    # Reshape to blocks
    arr_blocks = arr.reshape(nx, block_size, ny, block_size, nz, block_size)
    arr_blocks = arr_blocks.transpose(0,2,4,1,3,5)  # (nx,ny,nz,block,block,block)
    arr_blocks = arr_blocks.reshape(nx, ny, nz, -1) # (nx,ny,nz,block_size^3)
    # Take mode along last axis
    mode_vals, _ = mode(arr_blocks, axis=-1, keepdims=False)
    return mode_vals.squeeze()

def downsample_mode_2d(arr, block_size=7):
    sx, sy, sz = arr.shape
    if sz != 1:
        raise ValueError("This function expects z=1 for 2D models.")
    nx, ny = sx // block_size, sy // block_size
    arr = arr[:nx*block_size, :ny*block_size, :]
    arr_blocks = arr.reshape(nx, block_size, ny, block_size, 1)
    arr_blocks = arr_blocks.transpose(0,2,4,1,3)  # (nx,ny,1,block,block)
    arr_blocks = arr_blocks.reshape(nx, ny, -1) # (nx,ny,block_size^2)
    mode_vals, _ = mode(arr_blocks, axis=-1, keepdims=False)
    return mode_vals  # shape (nx, ny)

# Save A-scans to list
inputname = sys.argv[1]
name, extension = os.path.splitext(os.path.basename(inputname))
tracename = os.path.join(data_folder, name + ".out")

df = h5py.File(tracename, "r")
dset = df['rxs']
rx1 = dset['rx1']
trace = rx1['Ez']
signal = np.asarray(trace)

signal_stack_path = os.path.join(data_folder, 'signal_list_multi_2D_pep.npy')
try:
    signal_stack = np.load(signal_stack_path)
except FileNotFoundError:
    signal_stack = np.empty((0, signal.shape[0]))
signal_expanded = np.expand_dims(signal, axis=0)
signal_stack = np.concatenate([signal_stack, signal_expanded], axis=0)
np.save(signal_stack_path, signal_stack)

# Save permittivity model to list
permname = os.path.join(data_folder, name + "_permittivity.npy")
eps_r = np.load(permname)
perm = eps_r[2, :, :, 0].T  # shape (nx, ny)
print(f"Processing: {inputname}")
print(f"Perm shape: {perm.shape}, unique values: {np.unique(perm)}")
if perm.ndim == 2:
    perm = perm[:, :, np.newaxis]  # shape (nx, ny, 1)
perm_down = downsample_mode_2d(perm, block_size=7)
print(f"Downsampled perm shape: {perm_down.shape}, unique values: {np.unique(perm_down)}")
perm_stack_path = os.path.join(data_folder, 'perm_list_multi_2D_pep.npy')
try:
    perm_stack = np.load(perm_stack_path)
except FileNotFoundError:
    perm_stack = np.empty((0, *perm_down.shape))
perm_expanded = np.expand_dims(perm_down, axis=0)
perm_stack = np.concatenate([perm_stack, perm_expanded], axis=0)
np.save(perm_stack_path, perm_stack)