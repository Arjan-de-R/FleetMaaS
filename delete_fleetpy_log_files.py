import os
import sys

# Get the path to the directory where this script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# results folder
study_name = "competition_trb24"
results_folder = os.path.join(script_dir, "FleetPy", "studies", study_name, "results")

# scan results folder and delete log files
for sc in os.listdir(results_folder):
    sc_folder = os.path.join(results_folder, sc)
    if os.path.isdir(sc_folder) and os.path.isfile(os.path.join(sc_folder, "standard_eval.csv")):
        print("Found completed scenarios: " + sc + ". Deleting log files...")
        for log in os.listdir(sc_folder):
            if log.endswith(".log"):
                os.remove(os.path.join(sc_folder, log))
                print("  ... Deleted log file: " + os.path.join(sc_folder, log))