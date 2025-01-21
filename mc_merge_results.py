import os
import shutil

input_folder = "/Users/gregoire.szymanski/Documents/mc_raw_results"
output_folder = "/Users/gregoire.szymanski/Documents/mc_results"

identificator = '5s'

# Get the list of subfolders in the input folder
list_subfolders = [
    subfolder for subfolder in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, subfolder)) and subfolder.startswith("output_") and subfolder[7:].isdigit()
]

list_H = [0.1, 0.2, 0.3, 0.4, 0.5]

for H in list_H:
    filename = f"results{int(H * 10):02d}_{identificator}.txt"
    output_file_path = os.path.join(output_folder, filename)

    # Open the output file for writing
    with open(output_file_path, 'w') as output_file:
        for subfolder in list_subfolders:
            input_file_path = os.path.join(input_folder, subfolder, filename)

            # Check if the file exists in the current subfolder
            if os.path.exists(input_file_path):
                with open(input_file_path, 'r') as input_file:
                    # Copy content from input file to output file
                    output_file.write(input_file.read())

    # After successfully writing to the output file, delete the file in all subfolders
    # for subfolder in list_subfolders:
    #     input_file_path = os.path.join(input_folder, subfolder, filename)
    #     if os.path.exists(input_file_path):
    #         os.remove(input_file_path)

print("Processing complete.")
