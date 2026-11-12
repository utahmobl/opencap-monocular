import os

base_path = "/LabValidation_withVideos1"

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".avi") and "(trimmed)" in file:
            old_file_path = os.path.join(root, file)
            new_file_name = file.replace(" (trimmed)", "_trimmed")
            new_file_path = os.path.join(root, new_file_name)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} to {new_file_path}")
