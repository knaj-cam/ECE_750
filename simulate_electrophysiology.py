# author: Knajwa Cameron, 2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy
import random
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Patch

# Parameters for the model
ROWS, COLS = 40, 40
GENERATIONS = 300
AV_BLOCK_GENERATIONS = 500
HEART_RATE = 175
VENTRICULAR_HEART_RATE = 220 # slower than SA node pulse

# Cell Types
SA_NODE = -1
AV_NODE = -2
ATRIAL = -3
VENTRICULAR = -4
SCAR = -5

# Possible Cell States (Resting, Depolarized, Repolarized, Plateau, Scarred)
RESTING = 0
DEPOLARIZED = 1
REPOLARIZED = 2
PLATEAU = 3
SCARRED = 4

# Conduction Delays
AV_DELAY = 11 # Delay at AV node in generations

# Add heterogeneity across the cells in terms of how long they stay in each state
PLATEAU_PERIOD_MIN = ROWS
PLATEAU_PERIOD_MAX = ROWS + 2
REPOL_PERIOD_MIN = 1
REPOL_PERIOD_MAX = 2
DEPOL_PERIOD_MIN = 1
DEPOL_PERIOD_MAX = 3

DEPOLARIZATION_NEIGHBOUR_THRESHOLD = 1  # Threshold for depolarization (i.e. # of neighbours needed for a cell to be depolarized)

atria_boundary = int(ROWS * 0.25 ) # set atria as 25% of total heart cells
   
# Initialize the grid with all cells in the resting state
def initialize_grid(rows, cols):
    grid = np.zeros((rows, cols), dtype=object) # All cells are at RESTING state
    
    # Set type for atrial cells
    for r in range(atria_boundary):
        for c in range(cols):
            grid[r, c] = {'type': ATRIAL, 'state': RESTING}
    
    # Set type for ventricular cells
    for r in range(atria_boundary, rows):
        for c in range(cols):
            grid[r, c] = {'type': VENTRICULAR, 'state': RESTING}

    # Set SA node at a specific position, start as depolarized
    grid[0, 0] = {'type': SA_NODE, 'state': DEPOLARIZED}

    # Set AV node position
    grid[atria_boundary-1, cols-1] = {'type': AV_NODE, 'state': RESTING}

    return grid
    
def cell_info(grid, row, col, attribute):
    return grid[row, col][attribute]

# Create a custom colormap for the states
colors = ['#c7cdd6',  # RESTING (grey), 0
          '#FF0000',  # DEPOLARIZED (red), 1
          '#fcba03',  # REPOLARIZED (yellow), 2
          '#a68fdb',  # PLATEAU (purple), 3
          '#000000',  # SCARRED (black), 4
        ]

custom_cmap = ListedColormap(colors)

# Count the number of depolarized neighbors (Moore neighborhood)
def count_depolarized_neighbors(grid, row, col):
    depolarized_count = 0
    cell_type = cell_info(grid, row, col, "type")
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue # Skip the center cell
            r, c = (row + i), (col + j)
            if not (0 <= r < ROWS and 0 <= c < COLS): # make sure r and c are within boundaries
                continue

            same_type = check_neighbour_is_same_cell_type(cell_type, grid[r, c]["type"])

            if grid[r, c]["state"] == DEPOLARIZED and same_type:  # If the neighbor is depolarized and of the same cell type
                depolarized_count += 1
    
    return depolarized_count

def check_neighbour_is_same_cell_type(cell_type, neighbouring_cell_type):
    if cell_type == AV_NODE:
        return True if neighbouring_cell_type == ATRIAL else False
    elif neighbouring_cell_type == SA_NODE:
        return True if cell_type == ATRIAL else False
    elif neighbouring_cell_type == AV_NODE:
        return True if cell_type == VENTRICULAR else False
    else:
        return True if cell_type == neighbouring_cell_type else False

# Update the entire grid for one time step
def update_grid(grid, plateau_timers, plateau_periods, av_node_triggered, repol_timers, repol_periods, depol_timers, depol_periods):

    new_grid = deepcopy(grid)
    new_plateau_timers = plateau_timers.copy()
    new_repol_timers = repol_timers.copy()
    new_depol_timers = depol_timers.copy()

    # determine the new state for each cell
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            current_state = cell_info(grid, row, col, "state")
            cell_type = cell_info(grid, row, col, "type")

            # STATE TRANSITIONS
            if current_state == SCARRED:
                continue # skip scar cells
            elif current_state == RESTING:
                # A cell can depolarize if it has a count of depolarized neighbours greater than the threshold
                depolarized_neighbors = count_depolarized_neighbors(grid, row, col)

                if depolarized_neighbors >= DEPOLARIZATION_NEIGHBOUR_THRESHOLD:
                    if cell_type == AV_NODE:
                        if not av_node_triggered[0]:
                            av_node_triggered = [True, 0] # facilate AV delay, keep track of when AV node has recieved depolarization trigger
                    else:
                        new_grid[row][col]["state"] = DEPOLARIZED # other cells becomes depolarized immediately
                        new_depol_timers[row, col] += 1 # Start counting the depolarization period
                elif cell_type == AV_NODE and av_node_triggered[0]:
                    if av_node_triggered[1] >= AV_DELAY: # AV node can depolarized after a delay
                        new_grid[row][col]["state"] = DEPOLARIZED
                        av_node_triggered = [False, 0] # reset AV node values
                    else:
                        av_node_triggered = [True, av_node_triggered[1] + 1] # increase AV node delay counter

            elif current_state == DEPOLARIZED:
                # check if cell is past depol period, and increment counter
                if new_depol_timers[row, col] < depol_periods[row, col]:
                    new_depol_timers[row, col] += 1
                else:
                    # After the depolarization period ends, the cell goes to plateau state
                    new_grid[row][col]["state"] = PLATEAU # Start plateau period
                    new_plateau_timers[row, col] += 1 # Start counting the plateau period
                    new_depol_timers[row, col] = 0 # Reset the depolarization timer
            
            elif current_state == PLATEAU:
                if new_plateau_timers[row, col] < plateau_periods[row, col]:
                    new_plateau_timers[row, col] += 1
                else:
                    new_grid[row][col]["state"] = REPOLARIZED
                    new_repol_timers[row, col] += 1 # Start counting the repolarization period
                    new_plateau_timers[row, col] = 0 # Reset the plateau timer

            elif current_state == REPOLARIZED:
                # check if cell is past repol period, and increment counter
                if new_repol_timers[row, col] < repol_periods[row, col]:
                    new_repol_timers[row, col] += 1
                else:
                    # After the repolarization period ends, the cell goes to resting state
                    new_grid[row][col]["state"] = RESTING
                    new_repol_timers[row, col] = 0 # Reset the repolarization timer

    return new_grid, new_plateau_timers, av_node_triggered, new_repol_timers, new_depol_timers

def create_scar_tissue(grid, condition_params):
    scar_percentage = condition_params["scar_percentage"]
    scar_location = condition_params["scar_location"] #"random" or ((start_row, end_row), (start_col, end_col))

    # Add scar tissue based on percentage
    total_cells = grid.size
    num_scar_cells = int(total_cells * (scar_percentage/100))
    
    if scar_location=="random":
        scar_indices = np.random.choice(total_cells, num_scar_cells, replace=False)
        
        for idx in scar_indices:
            r, c = divmod(idx, grid.shape[1])
            grid[r, c] = {'type': SCAR, 'state': SCARRED} # Initialize scar cells
    else:
        (start_row, end_row), (start_col, end_col) = scar_location
        # Ensure the region is within grid bounds
        start_row, end_row = max(0, start_row), min(grid.shape[0], end_row)
        start_col, end_col = max(0, start_col), min(grid.shape[1], end_col)

        # Get all cell indices in the scar region
        scar_region_cells = [(r, c) for r in range(start_row, end_row)
                                      for c in range(start_col, end_col)]
        
        # Determine number of scar cells in the region based on scar percentage
        num_scar_cells = min(int(len(scar_region_cells) * (scar_percentage/100)), len(scar_region_cells))

        # Select cells in the defined region to be scar cells
        scar_indices = np.random.choice(len(scar_region_cells), num_scar_cells, replace=False)
        for idx in scar_indices:
            r, c = scar_region_cells[idx]
            if not cell_info(grid, r, c, "type") == AV_NODE:
                grid[r, c] = {'type': SCAR, 'state': SCARRED}
    
    return grid

# Check if the AV node is fully blocked by scar tissue
def is_av_node_blocked(grid, av_node_position):
    blocked = True
    directions = [(-1, 0), (0, -1), (-1, -1)] # Atrial neighbours: Up, left, diagonal left

    # For each AV node cell
    for (row, col) in av_node_position:
        
        # Check neighboring cells
        for dr, dc in directions:
            neighbor_row, neighbor_col = row + dr, col + dc

            # Ensure we stay within bounds of the grid
            if 0 <= neighbor_row < len(grid) and 0 <= neighbor_col < len(grid[0]):
                # If a neighboring cell is not scar tissue, the AV node is not fully blocked
                if grid[neighbor_row][neighbor_col].get('type') != SCAR:
                    blocked = False
                    break
    
    return blocked

# Randomly depolarizes cells in the atrial area to simulate atrial fibrillation.
def induce_atrial_fibrillation(grid, atrial_area, condition_params, depol_timers):
    prob_depolarize = condition_params.get("prob_depolarize", 5) # prob_depolarize: Probability that an atrial cell will depolarize randomly in a generation
    row_start, row_end = atrial_area
    max_depolarized_cells = 2
    depolarized_count = 0

    for r in range(row_start, row_end):
        for c in range(grid.shape[1]):
            cell_type = cell_info(grid, r, c, "type")
            current_state = cell_info(grid, r, c, "state")

            if cell_type == ATRIAL and current_state == RESTING:
                if random.random() < (prob_depolarize/100) and depolarized_count < max_depolarized_cells:
                    depolarized_count += 1
                    grid[r, c]['state'] = DEPOLARIZED
                    depol_timers[r, c] += 1
    
    return grid, depol_timers

# Main simulation function
def simulate_electrophysiology(condition="Healthy", condition_params={}, rows=ROWS, cols=COLS, generations=GENERATIONS, heart_rate=HEART_RATE, ventricular_heart_rate=VENTRICULAR_HEART_RATE):
    grid = initialize_grid(rows, cols)

    # initialize grid based on condition
    if condition=="Scarred":
        if condition_params["scar_location"] != "scarred":
            generations = AV_BLOCK_GENERATIONS 
        grid = create_scar_tissue(grid, condition_params)

    plateau_timers = np.zeros((rows, cols), dtype=int) # Timer for plateau periods
    plateau_periods = np.random.randint(PLATEAU_PERIOD_MIN, PLATEAU_PERIOD_MAX + 1, (rows, cols))
    depol_timers = np.zeros((rows, cols), dtype=int)  # Timer for depolarization periods
    depol_periods = np.random.randint(DEPOL_PERIOD_MIN, DEPOL_PERIOD_MAX + 1, (rows, cols))    
    repol_timers = np.zeros((rows, cols), dtype=int)  # Timer for repolarization periods    
    repol_periods = np.random.randint(REPOL_PERIOD_MIN, REPOL_PERIOD_MAX + 1, (rows, cols))

    for r in range(ROWS):
        for c in range(COLS):
            if cell_info(grid, r, c, "type") == VENTRICULAR:
                # Increase depolarization, repolarization and plateau period for ventricular cells for T wave effect
                repol_periods[r, c] = repol_periods[r, c] + random.randint(1, 2)
                depol_periods[r, c] = depol_periods[r, c] + random.randint(1, 2)
                plateau_periods[r, c] = plateau_periods[r, c] + random.randint(1, 2)

    av_node_triggered = [False, 0]
    ecg_data = [] # To store the ECG signal over time
    
    display_cell_types(grid)
    
    heart_model, ecg_line, generation_label = initial_plotting(rows, cols, generations)

    for generation in range(generations):
        # spontaneously generate a new pulse at a rate corresponding to the simulated heart reate
        if generation % heart_rate == 0:
            grid = spontaneous_depolarization(grid)

        # secondary ventricular pacemaker
        if is_av_node_blocked(grid, [(atria_boundary-1, cols-1)]) and generation!=0 and generation % ventricular_heart_rate == 0:
            grid = spontaneous_depolarization_ventricles(grid)
        
        # induce random fibrillation
        if condition=="Atrial Fibrillation":
            grid, depol_timers = induce_atrial_fibrillation(grid, (0, 20), condition_params, depol_timers)

        # update grid
        grid, plateau_timers, av_node_triggered, repol_timers, depol_timers = update_grid(grid, plateau_timers, plateau_periods, av_node_triggered, repol_timers, repol_periods, depol_timers, depol_periods) # Update the grid for the next generation

        # update Plot
        plotting_grid = create_plotting_grid(grid, rows, cols)

        # Update generation label
        generation_label.set_text(f"Generation: {generation}")

        # Update heart model plot
        heart_model.set_data(plotting_grid)

        # Measure the ECG signal at this time step
        ecg_signal = measure_ecg_signal(grid)
        smoothed_ecg = gaussian_filter(ecg_signal, sigma=100)
        ecg_data.append(smoothed_ecg)
        
        # Update ECG plot data
        ecg_line.set_data(range(len(ecg_data)), ecg_data)
        
        # Redraw the updated figure
        plt.pause(0.01) # Adjust pause duration as needed

    plt.show() # Show the final plot after simulation ends

# create a function to mimic spontaneous depolarization
def spontaneous_depolarization(grid):
    # corner cell of grid == SA node myocyte
    if grid[0, 0]["state"] != DEPOLARIZED:
        grid[0, 0]["state"] = DEPOLARIZED

    return grid

# if AV node is fully blocked by scar tissue, rely on the ventricle's secondary pacemaker
def spontaneous_depolarization_ventricles(grid):
    # AV node myocyte
    if grid[atria_boundary, COLS-1]["state"] != DEPOLARIZED:
        grid[atria_boundary, COLS-1]["state"] = DEPOLARIZED

    return grid

def display_cell_types(grid, rows=ROWS, cols=COLS):
    type_grid = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            if grid[row, col]["type"] == ATRIAL:
                type_grid[row, col] = 1
            elif grid[row, col]["type"] == VENTRICULAR:
                type_grid[row, col] = 2
            elif grid[row, col]["type"] == SA_NODE:
                type_grid[row, col] = 3
            elif grid[row, col]["type"] == AV_NODE:
                type_grid[row, col] = 4
            if grid[row, col]["type"] == SCAR:
                type_grid[row, col] = 5

    plt.imshow(type_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Cell Type (1: Atrial, 2: Ventricular, 3: SA Node, 4: AV Node, 5: Scar Tissue)")
    plt.title("Cell Types")
    plt.show()

def measure_ecg_signal(grid):
    ecg_signal = 0
     
    for row in range(ROWS):
        for col in range(COLS):
            cell_state = cell_info(grid, row, col, "state")
            cell_type = cell_info(grid, row, col, "type")


            if cell_state == DEPOLARIZED:
                # Atrial depolarization contributes to the P wave, scaled for atria
                if cell_type == ATRIAL:
                    ecg_signal += 0.5
                elif cell_type == VENTRICULAR:
                    # Ventricular depolarization contributes to the QRS complex, scaled for ventricles
                    ecg_signal += 1
            # Capture repolarization phase (T wave) for ventricular cells
            elif cell_state == REPOLARIZED and cell_type == VENTRICULAR:
                ecg_signal += 0.3
        
    return ecg_signal

def initial_plotting(rows, cols, generations):
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Initial heart model plot
    initial_plotting_grid = np.zeros((rows, cols), dtype=int)
    initial_plotting_grid[0,0] = DEPOLARIZED

    heart_model = ax1.imshow(initial_plotting_grid, cmap=custom_cmap, vmin=0, vmax=len(colors)-1, interpolation='nearest')
    ax1.set_title("Heart Model")
    ax1.axis('on')

    # Add the generation number label
    generation_label = ax1.text(0.5, 0.95, f"Generation: 0", ha='center', va='top', color='black', fontsize=12, transform=ax1.transAxes)

    # Create the legend for cell states
    legend_elements = [
        Patch(color='#c7cdd6', label='RESTING'),
        Patch(color='#FF0000', label='DEPOLARIZED'),
        Patch(color='#fcba03', label='REPOLARIZED'),
        Patch(color='#a68fdb', label='PLATEAU'),
        Patch(color="#000000", label='SCARRED')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=12, bbox_to_anchor=(1, 0.5), borderaxespad=0.)
    
    # Initial ECG plot
    ecg_line, = ax2.plot([], [], color='blue')
    ax2.set_xlim(0, generations)
    ax2.set_ylim(-20, 300) # cols*rows, Adjust as needed for signal range
    ax2.set_title("ECG Signal")
    ax2.set_xlabel("Time (Generations)")
    ax2.set_ylabel("Cell Count")
    ax2.grid(True)

    return heart_model, ecg_line, generation_label

def create_plotting_grid(grid, rows, cols):
    # seperate out the state values for plotting
    plotting_grid = np.zeros((rows, cols), dtype=int)
    for row in range(plotting_grid.shape[0]):
        for col in range(plotting_grid.shape[1]):
            plotting_grid[row, col] = cell_info(grid, row, col, "state")
    
    return plotting_grid

# Run the simulation with continuous plotting
# simulate_electrophysiology()
