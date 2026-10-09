import os
import sys
import numpy as np

in_folder = sys.argv[1]
dz = 0.005
block_size = 4
block_height = dz * block_size
num_blocks = int(4.0 / block_height)
soil_thickness = num_blocks * block_height  # Use this as your constant

heights_above_ground = []

for fname in sorted(os.listdir(in_folder)):
    if fname.endswith(".in"):
        rx_height = None
        with open(os.path.join(in_folder, fname), "r") as f:
            for line in f:
                if line.strip().startswith("#rx:"):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        rx_height = float(parts[2])
                    break
        if rx_height is not None:
            heights_above_ground.append(rx_height - soil_thickness)
        else:
            heights_above_ground.append(None)

print(heights_above_ground)

# Save as .npy in current directory, using the folder name in the file name
out_name = f"rx_heights_{os.path.basename(in_folder)}.npy"
np.save(out_name, np.array(heights_above_ground))