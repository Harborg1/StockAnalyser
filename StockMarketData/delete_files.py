import os
import shutil


"""Run VS code as adminstrator"""
""" In a terminal, copy: Set-ExecutionPolicy -ExecutionPolicy  Unrestricted -Scope Process"""

# Specify the directory to delete files from
del_dir_temp = r'C:\Users\cahar\AppData\Local\Temp'


def delete_files(dir):

    # List all files and directories in the specified temp directory
    for f in os.listdir(dir):
        # Create the full path for each file/directory
        full_path = os.path.join(dir, f)
        print(f"Processing: {full_path}")  # Print the file/directory being processed

        # Check if it is a file
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)  # Remove the file
                print("Removed file:", f, "from", dir)
            except Exception as e:
                print(f"Could not remove file {f}: {e}")
        # Check if it is a directory
        elif os.path.isdir(full_path):
            try:
                shutil.rmtree(full_path, ignore_errors=True)  # Remove the directory and its contents
                print("Removed directory:", f, "from", dir)
            except Exception as e:
                print(f"Could not remove directory {f}: {e}")

    print("Cleanup complete.")

delete_files(del_dir_temp)


