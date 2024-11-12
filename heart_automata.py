from testing_horizontal_boundary import simulate_electrophysiology
import PySimpleGUI as sg

# Define the layout for the GUI
layout = [
    [sg.Button("Healthy")],
    [sg.Button("Scarred (random location)")],
    [sg.Text("Scar Tissue Percentage (0 - 100):"), sg.InputText(key="-SCARPERCENTAGERANDOM-",size=(5, 1))],
    [sg.Button("Scarred (specific location):")],
    [sg.Text("Scar Tissue Dimensions:")],
    [sg.Text("start_row:"), sg.InputText(key="-STARTROW-", size=(5, 1)), sg.Text("end_row:"), sg.InputText(key="-ENDROW-", size=(5, 1)), sg.Text("start_col:"), sg.InputText(key="-STARTCOL-", size=(5, 1)), sg.Text("end_col:"), sg.InputText(key="-ENDCOL-", size=(5, 1))],
    [sg.Text("Scar Tissue Percentage (0 - 100):"), sg.InputText(key="-SCARPERCENTAGE-",size=(5, 1))],
    [sg.Button("Atrial Fibrillation")],
    [sg.Text("Probability of Depolarization (1-100):"), sg.InputText(key="-DEPOLPROB-",size=(5, 1))],
    [sg.Button("Close")],
    [sg.Text("Output:"), sg.Text("", size=(60, 1), key="-OUTPUT-")]
]

# Create the window
window = sg.Window("Modelling ElectroPhysiology of the Heart using Cellular Automata", layout)

# Event loop
while True:
    event, values = window.read()

    # Exit the loop if user closes the window
    if event == "Close" or event == sg.WINDOW_CLOSED:
        break
    elif event == "Healthy":
        window["-OUTPUT-"].update(f"Healthy Selected.")
        simulate_electrophysiology()
    elif event == "Scarred (specific location):":
        window["-OUTPUT-"].update(f"Scarred (specific location) selected.")
        scar_location= (
            (int(values['-STARTROW-']), int(values['-ENDROW-'])), 
            (int(values['-STARTCOL-']), int(values['-ENDCOL-']))
        )
        condition_params = {"scar_percentage": int(values['-SCARPERCENTAGE-']), "scar_location": scar_location} # random, or ((x1, x2), (y1, y2))
        
        # add validation that input boxes must be filled
        simulate_electrophysiology("Scarred", condition_params)
    elif event == "Scarred (random location)":
        condition_params = {"scar_percentage": int(values['-SCARPERCENTAGERANDOM-']), "scar_location": "random"} # random, or ((x1, x2), (y1, y2))        
        simulate_electrophysiology("Scarred", condition_params)
        window["-OUTPUT-"].update("Scarred (random location) selected.")
    elif event == "Atrial Fibrillation":
        condition_params = {"prob_depolarize": int(values['-DEPOLPROB-'])}   
        simulate_electrophysiology("Atrial Fibrillation", condition_params)
        window["-OUTPUT-"].update("Atrial Fibrillation selected.")


# Close the window
window.close()
