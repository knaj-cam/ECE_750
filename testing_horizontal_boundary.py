# author: Knajwa Cameron, 2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy
import random
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Patch

# Parameters for the model
ROWS, COLS = 50, 50 # 100 generations per heartbeat
GENERATIONS = 240
HEART_RATE = 120 # 20 generation pause btw each heart beat
VENTRICULAR_HEART_RATE = 160 # slower than SA node pulse

# Cell Types
SA_NODE = -1
AV_NODE = -2
ATRIAL = -3
VENTRICULAR = -4
SCAR = -5

# Possible Cell States (Resting, Depolarized, Repolarized, Refractory)
RESTING = 0
DEPOLARIZED = 1
REPOLARIZED = 2
REFRACTORY = 3
SCARRED = 4

# Conduction Delays
AV_DELAY = 11  # Delay at AV node in generations, in reality AV node delay 11% of time i.e. delay is 0.09s and Heart Beat is 0.8s

# Add heterogeneity across the cells in terms of how long they stay in each state
REFRACTORY_PERIOD_MIN = 4
REFRACTORY_PERIOD_MAX = 8
REPOL_PERIOD_MIN = 1
REPOL_PERIOD_MAX = 3
DEPOL_PERIOD_MIN = 1
DEPOL_PERIOD_MAX = 3

DEPOLARIZATION_NEIGHBOUR_THRESHOLD = 1  # Threshold for depolarization (i.e. # of neighbours needed for a cell to be depolarized)

# Parameters for FHN model
a = 0.7  # Recovery rate
b = 0.8  # Recovery sensitivity
tau = 12.5  # Time scaling for recovery variable
reset_threshold_v = 1.8  # Membrane potential threshold for resting transition
reset_threshold_w = 0.14  # Recovery variable threshold for resting transition
            
# Initialize the grid with all cells in the resting state
def initialize_grid(rows, cols):
    grid = np.zeros((rows, cols), dtype=object) # All cells are at RESTING state (0)

    atria_boundary = int(ROWS * 0.4 ) # atria is 40% of total heart cells
    
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
    grid[atria_boundary, cols-1] = {'type': AV_NODE, 'state': RESTING}

    return grid
    
def cell_info(grid, row, col, attribute):
    return grid[row, col][attribute]

# Create a custom colormap for the states
colors = ['#c7cdd6',  # RESTING (grey), 0
          '#FF0000',  # DEPOLARIZED (red), 1
          '#fcba03',  # REPOLARIZED (yellow), 2
          '#a68fdb',  # REFRACTORY (purple), 3
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
def update_grid(grid, refractory_timers, v, w, refractory_periods, av_node_triggered, repol_timers, repol_periods, depol_timers, depol_periods):

    new_grid = deepcopy(grid)
    new_v, new_w = v.copy(), w.copy()
    new_refractory_timers = refractory_timers.copy()
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
                    if cell_type == AV_NODE and not av_node_triggered[0]:
                        av_node_triggered = [True, 0] # facilate AV delay, keep track of when AV node has recieved depolarization trigger
                    else:
                        new_grid[row][col]["state"] = DEPOLARIZED # other cells becomes depolarized immediately
                        new_depol_timers[row, col] = 1  # Start counting the depolarization period
                elif cell_type == AV_NODE and av_node_triggered[0] == True:
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
                    new_grid[row][col]["state"] = REPOLARIZED
                    new_repol_timers[row, col] = 1  # Start counting the repolarization period
                    new_depol_timers[row, col] = 0  # Reset the depolarization timer
                
            elif current_state == REPOLARIZED:
                # check if cell is past repol period, and increment counter
                if new_repol_timers[row, col] < repol_periods[row, col]:
                    new_repol_timers[row, col] += 1
                else:
                    # After the repolarization period ends, the cell goes to refractory state
                    new_grid[row][col]["state"] = REFRACTORY # Start refractory period
                    new_refractory_timers[row, col] = 1 # Start counting the refractory period
                    new_repol_timers[row, col] = 0  # Reset the repolarization timer

            elif current_state == REFRACTORY:
                # check if cell is past refractory period, and increment counter
                if new_refractory_timers[row, col] < refractory_periods[row, col]:
                    new_refractory_timers[row, col] += 1
                else:
                    # After the refractory period ends, the cell returns to the resting state and can be excited again
                    new_grid[row][col]["state"] = RESTING
                    new_refractory_timers[row, col] = 0  # Reset the refractory timer

    return new_grid, new_refractory_timers, new_v, new_w, av_node_triggered, new_repol_timers, new_depol_timers

# Initialize FHN variables for each cell
def initialize_fhn(rows, cols):
    v = np.zeros((rows, cols))  # Membrane potential
    w = np.zeros((rows, cols))  # Recovery variable
    return v, w

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
            grid[r, c] = {'type': SCAR, 'state': SCARRED}  # Initialize scar cells
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
            grid[r, c] = {'type': SCAR, 'state': SCARRED}
    
    return grid

def is_av_node_blocked(grid, av_node_position):
    """
    Checks if the AV node is fully blocked by scar tissue.

    Parameters:
    - grid: The cardiac grid
    - av_node_position: Tuple for the AV node position (row, col) or list of positions for larger AV nodes.

    Returns:
    - True if the AV node is fully blocked by scar tissue, otherwise False.
    """
    blocked = True
    directions = [(-1, 0), (-1, -1)] # Atrial neighbours: Up, diagonal left

    # For each AV node cell
    for (row, col) in av_node_position:
        # Check if the AV node cell itself is scar tissue
        if grid[row][col].get('type') == SCAR:
            return True
        
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

# FIX add more rules for setting to depolarized, maybe has to still have depolarized neighbours?
def induce_atrial_fibrillation(grid, atrial_area, condition_params): # 0.05
    """
    Randomly depolarizes cells in the atrial area to simulate atrial fibrillation.
    
    Parameters:
    - grid: The cardiac grid.
    - atrial_area: Tuple defining the atrial region (row_start, row_end).
    - prob_depolarize: Probability that an atrial cell will depolarize randomly each timestep.
    """
    prob_depolarize = condition_params.get("prob_depolarize", 5)
    row_start, row_end = atrial_area

    # maybe set a maximum on how many cells can be depolarized at once
    for r in range(row_start, row_end):
        for c in range(grid.shape[1]):
            if grid[r, c]['type'] == ATRIAL and grid[r, c]['state']==RESTING:
                if random.random() < (prob_depolarize/100) and count_depolarized_neighbors(grid, r, c) > 2:
                    #if random.random() < 0.03:  # Random chance to prevent continuous firing
                    grid[r, c]['state'] = DEPOLARIZED
    
    return grid

# Main simulation function
def simulate_electrophysiology(condition="Healthy", condition_params={}, rows=ROWS, cols=COLS, generations=GENERATIONS, heart_rate=HEART_RATE, ventricular_heart_rate=VENTRICULAR_HEART_RATE):
    # initialize grid based on condition
    grid = initialize_grid(rows, cols)

    if condition=="Scarred":
        grid = create_scar_tissue(grid, condition_params)

    # could set the repol timer for ventricular cells to be a higher random number
    refractory_timers = np.zeros((rows, cols), dtype=int)  # Timer for refractory periods
    refractory_periods = np.random.randint(REFRACTORY_PERIOD_MIN, REFRACTORY_PERIOD_MAX + 1, (rows, cols))
    repol_timers = np.zeros((rows, cols), dtype=int)  # Timer for repolarization periods
    repol_periods = np.random.randint(REPOL_PERIOD_MIN, REPOL_PERIOD_MAX + 1, (rows, cols))
    depol_timers = np.zeros((rows, cols), dtype=int)  # Timer for depolarization periods
    depol_periods = np.random.randint(DEPOL_PERIOD_MIN, DEPOL_PERIOD_MAX + 1, (rows, cols))
    
    v, w = initialize_fhn(rows, cols)
    av_node_triggered = [False, 0]
    ecg_data = []  # To store the ECG signal over time
    
    display_cell_types(grid)
    
    heart_model, ecg_line, generation_label = initial_plotting(rows, cols, generations)

    for generation in range(generations):
        # spontaneously generate a new pulse at a rate corresponding to the simulated heart reate
        if generation % heart_rate == 0:
            grid = spontaneous_depolarization(grid)

        # secondary ventricular pacemaker
        if is_av_node_blocked(grid, [(20,49)]) and generation!=0 and generation % ventricular_heart_rate == 0:
            grid = spontaneous_depolarization_ventricles(grid)
        
        # induce random fibrillation, can also have it be period?  if generation % af_induce_interval == 0:
        if condition=="Atrial Fibrillation":
            grid = induce_atrial_fibrillation(grid, (0, 20), condition_params)
        # if condition=="Scarred":
        #     # Scar tissue can also contribute to arrhythmias, such as atrial fibrillation, where the atria depolarize randomly and chaotically. You've already implemented a function induce_atrial_fibrillation that depolarizes atrial cells randomly. This could be triggered more frequently in the presence of scar tissue to simulate arrhythmic behavior following a heart attack.
        #     #set depol prob based on percentage of scar tissue
        #     grid = induce_atrial_fibrillation(grid, (0, 20), {"prob_depolarize": 10})

        # update grid
        grid, refractory_timers, v, w, av_node_triggered, repol_timers, depol_timers = update_grid(grid, refractory_timers, v, w, refractory_periods, av_node_triggered, repol_timers, repol_periods, depol_timers, depol_periods) # Update the grid for the next generation

        # update Plot

        plotting_grid = create_plotting_grid(grid, rows, cols)

        # Update generation label
        generation_label.set_text(f"Generation: {generation}")

        # Update heart model plot
        heart_model.set_data(plotting_grid)

        # Measure the ECG signal at this time step
        ecg_signal = measure_ecg_signal(grid, generation)
        ecg_data.append(ecg_signal)
        # Update ECG plot data
        ecg_line.set_data(range(len(ecg_data)), ecg_data)
        
        # Redraw the updated figure
        plt.pause(0.01) # Adjust pause duration as needed

    plt.show()  # Show the final plot after simulation ends

# create a function to mimic spontaneous depolarization
def spontaneous_depolarization(grid):
    # at rest the SA nodal myocytes depolarize at an intrinsic rate between 60-100 beats per minutes
    # corner cell of grid == SA node myocyte
    if grid[0, 0]["state"] != DEPOLARIZED:
        grid[0, 0]["state"] = DEPOLARIZED

    return grid

# if AV node is fully blocked by scar tissue, rely on the ventricle's secondary pacemaker (slower heart beat, e.g. 30-40bpm)
def spontaneous_depolarization_ventricles(grid):
    # AV node myocyte
    if grid[20, 49]["state"] != DEPOLARIZED: # can i search for cell with type AV node?
        grid[20, 49]["state"] = DEPOLARIZED

    return grid

# Update FHN dynamics for a single time step
def update_fhn(v, w, current_state):
    dvdt = v - (v**3) / 3 - w + (1 if current_state == REFRACTORY else 0)
    dwdt = (v + a - b * w) / tau
    v += dvdt
    w += dwdt

    # Apply damping to ensure that variables decrease smoothly
    v *= 0.9
    w *= 0.9
    return v, w

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
    plt.show()

def measure_ecg_signal(grid, generation, qrs_interval=1):
    """
    Measure ECG signal by selectively adding depolarization states.
    Parameters:
        grid (2D array): The current grid of cell states.
        generation (int): The current generation (time step).
        qrs_interval (int): Interval after which ventricular depolarization contributes to the ECG.
    Returns:
        ecg_signal (float): The simulated ECG signal at this time step.
    """
    ecg_signal = 0
    
    # Measure atrial depolarization continuously to simulate the P wave
    for row in range(int(grid.shape[0] * 0.4)):  # Atria
        for col in range(grid.shape[1]):
            if cell_info(grid, row, col, "state") == DEPOLARIZED:
                ecg_signal += 0.5#,  Weight for atrial contribution (adjust as needed)

    # Measure ventricular depolarization at specific intervals to simulate the QRS complex
    if generation % qrs_interval == 0:
        for row in range(int(grid.shape[0] * 0.4), grid.shape[0]):  # ventricles
            for col in range(grid.shape[1]):
                if cell_info(grid, row, col, "state") == DEPOLARIZED: # ( or REPOLARIZED):
                    ecg_signal += 0.5 #, Heavier weight for ventricular contribution (adjust as needed)

    return ecg_signal

def initial_plotting(rows, cols, generations):
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Initial heart model plot
    initial_plotting_grid = np.zeros((rows, cols), dtype=int)
    initial_plotting_grid[0,0] = DEPOLARIZED # clean this up

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
        Patch(color='#a68fdb', label='REFRACTORY'),
        Patch(color="#000000", label='SCARRED')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=12, bbox_to_anchor=(1, 0.5), borderaxespad=0.)
    
    # Initial ECG plot
    ecg_line, = ax2.plot([], [], color='blue')
    ax2.set_xlim(0, generations)
    ax2.set_ylim(-20, 200) # cols*rows, Adjust as needed for signal range
    ax2.set_title("Simulated ECG Signal")
    ax2.set_xlabel("Time (Generations)")
    ax2.set_ylabel("ECG Signal (Depolarized Cell Count)")
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
#simulate_electrophysiology()
