import glob
import os

# Define the directory you want to search
search_dir = "C:\\Program Files"  # Replace with the directory you want to search

# Search for all .exe files in the directory and its subdirectories
exe_files = glob.glob(os.path.join(search_dir, '**', '*.exe'), recursive=False)

list=list[exe_files]

# Print the list of found .exe files
for k  in list:
    print(k)

# print(list)