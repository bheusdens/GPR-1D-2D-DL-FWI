import random
import sys
import textwrap
import os

dz = 0.003
block_size = 7
block_height = dz * block_size  # 0.021 m
num_sims = int(sys.argv[1])

for j in range(num_sims):
    # Total soil thickness (e.g., 4 m)
    total_thickness = block_height * int(4.0 / block_height)

    # Randomly choose number of layers (e.g., 1 to 4)
    n_layers = random.randint(1, 4)

    # Randomly split total_thickness into n_layers
    if n_layers == 1:
        layer_thicknesses = [total_thickness]
    else:
        breaks = sorted(random.sample(range(1, 1000), n_layers - 1))
        layer_thicknesses = [b - a for a, b in zip([0] + breaks, breaks + [1000])]
        layer_thicknesses = [total_thickness * (x / 1000) for x in layer_thicknesses]

    Mats = ""
    boxes = ""
    y_top = 0.0
    for i, thickness in enumerate(layer_thicknesses):
        # Randomize Peplinski soil parameters for each layer
        sand = round(random.uniform(0.1, 0.5), 2)
        clay = round(random.uniform(0.1, 0.3), 2)
        bulk = round(random.uniform(1.1, 1.6), 2)
        particle = round(random.uniform(2.6, 2.7), 2)
        conductivity = 0.001
        moisture = round(random.uniform(0.01, 0.5), 2)

        mat_name = f"my_soil{i}"
        box_name = f"my_soil_box{i}"

        Mats += f"#soil_peplinski: {sand} {clay} {bulk} {particle} {conductivity} {moisture} {mat_name}\n"
        boxes += (
            f"#fractal_box: 0 {y_top:.3f} 0 1.002 {y_top + thickness:.3f} 0.003 "
            f"{thickness:.3f} 1 1 1 50 {mat_name} {box_name}\n"
        )
        y_top += thickness

    # Randomize dipole and receiver height between 6 and 13.8 m, snapped to block height
    dipole_rx_height = random.uniform(6.0, 13.8)
    dipole_rx_height = round(dipole_rx_height / block_height) * block_height

    Constants = textwrap.dedent(f"""
    #domain: 1.002 14.0 0.003
    #dx_dy_dz: 0.003 0.003 0.003
    #time_window: 3e-7

    #waveform: ricker 1 825e6 my_ricker
    #hertzian_dipole: z 0.45 {dipole_rx_height:.3f} 0.0015 my_ricker
    #rx: 0.55 {dipole_rx_height:.3f} 0.0015

    """)

    title = "#title: Multi Layer Peplinski Model Simulation\n"
    geom = f"#geometry_view: 0 0 0 1.002 14.0 0.003 0.003 0.003 0.003 geom_multi n\n"

    output_folder = "2D_Data_Gen_Pep_Multi"
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"Simulation_{j}_2D_Pep.in")
    with open(filename, "w") as f:
        f.write(title)
        f.write(Constants)
        f.write(Mats)
        f.write(boxes)
        f.write(geom)

    print(f"Simulation {j}: layers={n_layers}, thicknesses={layer_thicknesses}, dipole/rx height={dipole_rx_height:.3f}")