import os
import re
import shutil

# Set your base directory where all studies are stored
BASE_DIRECTORY = "../../Fleetpy/studies"  # Change this to your actual base folder

def clean_old_folders(parent_dir):
    pattern = re.compile(r"^(.*?)-day-(\d+)$")
    folder_map = {}

    # Scan all subfolders
    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)
        if os.path.isdir(folder_path):
            match = pattern.match(folder)
            if match:
                base_name, day_number = match.groups()
                day_number = int(day_number)

                if base_name not in folder_map or day_number > folder_map[base_name]:
                    folder_map[base_name] = day_number

    # Remove files within selected folders
    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)
        match = pattern.match(folder)
        if match:
            base_name, day_number = match.groups()
            if int(day_number) < folder_map[base_name]:
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        # You can adjust this condition based on the filenames you want to delete
                        # For now, let's delete files that match a certain pattern (e.g., "specific_file")
                        if file in files_to_remove:
                            # print(f"Deleting file: {file_path}")
                            # Uncomment the next line to actually delete the file
                            os.remove(file_path)
                        # else:
                            # print(f"Keeping file: {file_path}")
                # If you want to remove the entire folder
                # print(f"Deleting: {folder_path}")
                # # shutil.rmtree(folder_path)
                # print(f"Would delete: {folder_path}")
            else:
                print(f"Keeping: {folder_path}")

if __name__ == "__main__":
    # List of filenames to remove
    files_to_remove = [
        "1_user-stats.csv",
        "2-2_op-stats.csv",
        "00_simulation.log"  # Add your specific filenames here
    ]

    study_name = input("Enter the study name: ").strip()
    parent_directory = os.path.join(BASE_DIRECTORY, study_name, "results")

    if not os.path.exists(parent_directory):
        print(f"Error: The folder '{parent_directory}' does not exist.")
    else:
        clean_old_folders(parent_directory)
