import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Patch
import matplotlib.colors

# Parameters for the model
ROWS, COLS = 50, 50
GENERATIONS = 240
HEART_RATE = 120 # 1 beat == time it takes for signal to traverse entire grid

# Cell Types
SA_NODE = -1
AV_NODE = -2
ATRIAL = -3
VENTRICULAR = -4

# Possible Cell States (Resting, Depolarized, Repolarized, Refractory)
RESTING = 0
DEPOLARIZED = 1
REPOLARIZED = 2
REFRACTORY = 3

# Conduction Delays
AV_DELAY = 10  # Delay at AV node in generations, add spac ein the ECG btw atria pulse and ventricular peak

# Refractory period should last for a period of time
REFACTORY_PERIOD = 8 # Length of refractory period (as number of generations)
REFRACTORY_PERIOD_MIN = 3
REFRACTORY_PERIOD_MAX = 6
DEPOL_PERIOD_MIN = 2
DEPOL_PERIOD_MAX = 4
VENTRICULAR_REPOLARIZATION_DELAY = 1

THRESHOLD = 1  # Threshold for depolarization (i.e. # of neighbours needed for a cell to be depolarized)

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
    
    # Set atrial cells, loop to ensure dictionaries are unique objects in each cell
    for r in range(atria_boundary):
        for c in range(cols):
            grid[r, c] = {'type': ATRIAL, 'state': RESTING}
    
    # Set ventricular cell
    for r in range(atria_boundary, rows):
        for c in range(cols):
            grid[r, c] = {'type': VENTRICULAR, 'state': RESTING}

    # Set SA node at a specific position
    grid[0, 0] = {'type': SA_NODE, 'state': DEPOLARIZED}  # start with SA node depolarize
    # Set AV node position
    grid[atria_boundary, cols-1] = {'type': AV_NODE, 'state': RESTING}

    return grid

# Count the number of depolarized neighbors (Moore neighborhood)
def count_depolarized_neighbors(grid, row, col):
    depolarized_count = 0
    cell_type = grid[row, col]["type"] # make this a method

    # boundary conditions
    rows, cols = grid.shape
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue # Skip the center cell
            r, c = (row + i), (col + j)
            if not (0 <= r < rows and 0 <= c < cols): # make sure r and c are within boundaries
                continue
            #print(f"count depol, cell_type: {cell_type} and r: {r} and c: {c} and state: {grid[r, c]["state"]} and type: {grid[r, c]["type"]}")

            same_type = check_repol_same_cell_type(cell_type, grid[r, c]["type"])
            # print(f"same type?: {same_type}")
            if grid[r, c]["state"] == DEPOLARIZED and same_type:  # If the neighbor is depolarized and of the same cell type
                depolarized_count += 1
    return depolarized_count

def check_repol_same_cell_type(cell_type, neighbouring_cell_type):
    #print(f"checking cell type, cell_type: {cell_type} and neighbour: {neighbouring_cell_type}")
    if cell_type == AV_NODE:
        return True if neighbouring_cell_type == ATRIAL else False
    elif neighbouring_cell_type == SA_NODE:
        return True if cell_type == ATRIAL else False
    elif neighbouring_cell_type == AV_NODE:
        return True if cell_type == VENTRICULAR else False
    else:
        return True if cell_type == neighbouring_cell_type else False

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
            if grid[row, col]["state"] == DEPOLARIZED:
                ecg_signal += 0.5#,  Weight for atrial contribution (adjust as needed)

    # Measure ventricular depolarization at specific intervals to simulate the QRS complex
    if generation % qrs_interval == 0:
        for row in range(int(grid.shape[0] * 0.4), grid.shape[0]):  # ventricles
            for col in range(grid.shape[1]):
                if grid[row, col]["state"] == DEPOLARIZED: # ( or REPOLARIZED):
                    ecg_signal += 0.5 #, Heavier weight for ventricular contribution (adjust as needed)

    # Measure ventricular repolarization to simulate the T wave
    # for row in range(int(grid.shape[0] * 0.4), grid.shape[0]):  # Ventricles
    #     for col in range(grid.shape[1]):
    #         if grid[row, col]["state"] == REPOLARIZED:
    #             ecg_signal += 1.2  # Lighter weight for repolarization (adjust for T wave visibility)
    return ecg_signal

# Measure the ECG signal by summing the depolarized states at certain electrode points
# def measure_ecg_signal(grid, generation):
#     # Simulating electrodes placed at certain points (e.g., corners or center of the grid)
#     electrodes = [(0, 0), (0, COLS-1), (ROWS-1, 0), (ROWS-1, COLS-1)]  # electrode positions at corners
#     ecg_signal = 0
#     for (r, c) in electrodes:
#         # Add contribution to ECG signal only from depolarized cells
#         ecg_signal += np.sum(grid == DEPOLARIZED)
#         #ecg_signal += np.sum(grid == DEPOLARIZED)  # Count all depolarized cells
#     return ecg_signal

# Update the entire grid for one time step
def update_grid(grid, refractory_timers, v, w, refractory_periods, vent_repol_timers, av_node_triggered, repol_timers, repol_periods):
    # When initializing new_grid, ensure that each dictionary is copied independently. This avoids the issue where all cells in a row or column reference the same dictionary.
    new_grid = deepcopy(grid) # copy the grid, shallow copy doesn't work because any nested structure (like dictionaries inside the grid) will still refer to the original objects, i.e. they will be overwritten in my loop below - deep copy creates a fully independent copy of the entrure structure so modying new_grid doesn't affect grid
    new_v, new_w = v.copy(), w.copy()
    new_refractory_timers = refractory_timers.copy()  # Copy the refractory timers
    new_vent_repol_timers = vent_repol_timers.copy()  # Copy ventricular repolarization timers
    new_repol_timers = repol_timers.copy()

    # determine the new state for each cell
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            current_state = grid[row, col]["state"]
            cell_type = grid[row, col]["type"]
            
            # STATE TRANSITIONS
            if current_state == RESTING: # If the cell is at resting state
                # A cell can depolarize if it has a count of depolarized neighbours greater than the threshold
                depolarized_neighbors = count_depolarized_neighbors(grid, row, col)

                if depolarized_neighbors >= THRESHOLD:  # Threshold number of depolarized neighbors
                    if cell_type == AV_NODE:
                        # The AV node also plays a crucial role in delaying the electrical signal before it is transmitted to the ventricles. This delay ensures that the atria have enough time to contract fully before the ventricles are activated.
                        av_node_triggered = ["true", 0]
                    else:
                        new_grid[row, col]["state"] = DEPOLARIZED  # Cell becomes depolarized
                elif cell_type == AV_NODE and av_node_triggered[0] == "true":
                    if av_node_triggered[1] >= AV_DELAY:
                        new_grid[row, col]["state"] = DEPOLARIZED
                        av_node_triggered = ["false", 0]
                    else:
                        av_node_triggered = ["true", av_node_triggered[1] + 1]

            elif current_state == DEPOLARIZED:  # If the cell is depolarized
                # Handle ventricular cells with a repolarization delay
                if cell_type == VENTRICULAR:
                    if new_vent_repol_timers[row, col] < VENTRICULAR_REPOLARIZATION_DELAY:
                        new_vent_repol_timers[row, col] += 1  # Increment repolarization timer
                    else:
                        new_grid[row, col]["state"] = REPOLARIZED
                        new_vent_repol_timers[row, col] = 0  # Reset the repolarization timer
                        new_repol_timers[row, col] = 1  # Start counting the repolarization period
                else:
                    # Non-ventricular cells repolarize immediately
                    new_grid[row, col]["state"] = REPOLARIZED
                    new_repol_timers[row, col] = 1  # Start counting the repolarization period
                
            elif current_state == REPOLARIZED:  # If the cell is repolarized
                # Cells enter refractory period after repolarization
                # Increment the repolarization timer
                if new_repol_timers[row, col] < repol_periods[row, col]: # into the repolarization period for each cell
                    new_repol_timers[row, col] += 1
                else:
                    # After the repolarization period ends, the cell goes to refractory state
                    new_grid[row, col]["state"] = REFRACTORY  # Start refractory period
                    new_refractory_timers[row, col] = 1  # Start counting the refractory period
                    new_repol_timers[row, col] = 0  # Reset the repolarization timer

            elif current_state == REFRACTORY:  # Refractory period
                # Transition back to RESTING if below FHN reset thresholds
                # new_v[row, col], new_w[row, col] = update_fhn(v[row, col], w[row, col], current_state)

                # print(f"VALUE: {new_w[row, col]} and THRESH: {reset_threshold_w} and result: {new_w[row, col] < reset_threshold_w}")
                # if new_w[row, col] < reset_threshold_w: #new_v[row, col] < reset_threshold_v and
                #     print(f"RREST")
                #     new_grid[row, col]["state"] = RESTING

                # Increment the refractory timer
                if new_refractory_timers[row, col] < refractory_periods[row, col]: #REFACTORY_PERIOD: # introduce randomness into the refractpry period for each cell
                    new_refractory_timers[row, col] += 1
                else:
                    # After the refractory period ends, the cell returns to the resting state and can be excited again
                    new_grid[row, col]["state"] = RESTING
                    new_refractory_timers[row, col] = 0  # Reset the refractory timer

    return new_grid, new_refractory_timers, new_v, new_w, new_vent_repol_timers, av_node_triggered, new_repol_timers

# Create a custom colormap for the states
colors = ['#c7cdd6',  # RESTING (grey), 0
          '#FF0000',  # DEPOLARIZED (red), 1
          '#fcba03',  # REPOLARIZED (yellow), 2
          '#a68fdb']  # REFRACTORY (purple), 3

#custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue", "green"])
custom_cmap = ListedColormap(colors)

# Initialize FHN variables for each cell
def initialize_fhn(rows, cols):
    v = np.zeros((rows, cols))  # Membrane potential
    w = np.zeros((rows, cols))  # Recovery variable
    return v, w

# Main simulation function
def simulate_electrophysiology(rows=ROWS, cols=COLS, generations=GENERATIONS, heart_rate=HEART_RATE):
    grid = initialize_grid(rows, cols)
    
    # Create a grid plot where atrial = 1, ventricular = 2, SA = 3, AV = 4
    display_cell_types(grid)

    refractory_timers = np.zeros((rows, cols), dtype=int)  # Timer for refractory periods
    refractory_periods = np.random.randint(REFRACTORY_PERIOD_MIN, REFRACTORY_PERIOD_MAX + 1, (rows, cols))
    repol_timers = np.zeros((rows, cols), dtype=int)  # Timer for depolarization
    repol_periods = np.random.randint(DEPOL_PERIOD_MIN, DEPOL_PERIOD_MAX + 1, (rows, cols))
    v, w = initialize_fhn(rows, cols)
    vent_repol_timers = np.zeros((rows, cols), dtype=int)
    av_node_triggered = ["false", 0]
    ecg_data = []  # To store the ECG signal over time
    
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    # Initial heart model plot
    initial_plotting_grid = np.zeros((rows, cols), dtype=int)
    initial_plotting_grid[0,0] = DEPOLARIZED # clean this up

    # When plotting, set vmin=0 and vmax=len(colors)-1
    # This ensures that imshow interprets the values in grid as indices from 0 to len(colors) - 1, aligning with the colors in ListedColormap.
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
        Patch(color='#a68fdb', label='REFRACTORY')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=12, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Initial ECG plot
    ecg_line, = ax2.plot([], [], color='blue')
    ax2.set_xlim(0, generations)
    ax2.set_ylim(-20, 80) # cols*rows, Adjust as needed for signal range
    ax2.set_title("Simulated ECG Signal")
    ax2.set_xlabel("Time (Generations)")
    ax2.set_ylabel("ECG Signal (Depolarized Cell Count)")
    ax2.grid(True)

    for generation in range(generations):
        # spontaneously generate a new pulse at a rate corresponding to the simulated heart reate
        if generation % heart_rate == 0:
            grid = spontaneous_depolarization(grid)

        grid, refractory_timers, v, w, vent_repol_timers, av_node_triggered, repol_timers  = update_grid(grid, refractory_timers, v, w, refractory_periods, vent_repol_timers, av_node_triggered, repol_timers, repol_periods) # Update the grid for the next generation

        # Update generation label
        generation_label.set_text(f"Generation: {generation}")

        # seperate out the state values for plotting
        plotting_grid = np.zeros((ROWS, COLS), dtype=int)
        for row in range(plotting_grid.shape[0]):
            for col in range(plotting_grid.shape[1]):
                plotting_grid[row, col] = grid[row, col]["state"]

        # Measure the ECG signal at this time step and store it .. could update to display continously
        ecg_signal = measure_ecg_signal(grid, generation)
        ecg_data.append(ecg_signal)

        # Apply a smoothing filter to ECG signal for realism
        # smoothed_ecg = gaussian_filter1d(ecg_data, sigma=2)
        
        # Update heart model plot
        heart_model.set_data(plotting_grid)

        # Update ECG plot data
        #ecg_line.set_data(range(len(smoothed_ecg)), smoothed_ecg)
        ecg_line.set_data(range(len(ecg_data)), ecg_data)
        
        # Redraw the updated figure
        # In your simulation loop, set the pause to match one generation = 1 heartbeat:
        plt.pause(0.01) # Adjust pause duration as needed


    plt.show()  # Show the final plot after simulation ends

# create a function to mimic spontaneous depolarization
def spontaneous_depolarization(grid):
    # at rest the SA nodal myocytes depolarize at an intrinsic rate between 60-100 beats per minutes
    # corner cell of grid == SA node myocyte
    if grid[0, 0]["state"] != DEPOLARIZED:
        grid[0, 0]["state"] = DEPOLARIZED
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

def display_cell_types(grid, rows = ROWS, cols=COLS):
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

    plt.imshow(type_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Cell Type (1: Atrial, 2: Ventricular, 3: SA Node, 4: AV Node)")
    plt.show()

# Run the simulation with continuous plotting
simulate_electrophysiology()
