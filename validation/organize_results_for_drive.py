import os
import shutil
import glob

# Destination directory for the organized results
DEST_DIR = "organized_results_for_drive"


def find_trial_processing_path(path):
    """
    Finds the correct path to process a trial, looking for an 'OpenSim' directory.
    This handles cases where there's a nested directory with the same name.
    """
    if "OpenSim" in os.listdir(path) and os.path.isdir(os.path.join(path, "OpenSim")):
        return path

    # Check one level deeper for nested directories.
    for item in os.listdir(path):
        new_path = os.path.join(path, item)
        if os.path.isdir(new_path):
            try:
                if "OpenSim" in os.listdir(new_path) and os.path.isdir(
                    os.path.join(new_path, "OpenSim")
                ):
                    return new_path
            except PermissionError:
                continue  # Ignore directories we can't read

    return None


def find_and_copy_files(trial_path, dest_trial_path):
    """
    Finds and copies the model, IK, synced IK, and lag files for a given trial.
    """
    trial_name = os.path.basename(os.path.normpath(dest_trial_path))

    # 1. Find and copy model.osim
    model_search_path = os.path.join(
        trial_path, "OpenSim", "Model", "*", "LaiUhlrich2022_scaled_no_patella.osim"
    )
    model_paths = glob.glob(model_search_path)
    if model_paths:
        shutil.copy(model_paths[0], os.path.join(dest_trial_path, "model.osim"))
    else:
        print(f"    - model.osim not found for trial {trial_name}")

    # 2. Find and copy ik.mot (non-synced)
    ik_path_found = None
    ik_base_path = os.path.join(trial_path, "OpenSim", "IK")
    if os.path.isdir(ik_base_path):
        for item in os.listdir(ik_base_path):
            item_path = os.path.join(ik_base_path, item)
            # Look in subdirectories of IK, but explicitly exclude 'shiftedIK'.
            if os.path.isdir(item_path) and item != "shiftedIK":
                mot_files = glob.glob(os.path.join(item_path, "*.mot"))
                if mot_files:
                    ik_path_found = mot_files[0]
                    break
    if ik_path_found:
        shutil.copy(ik_path_found, os.path.join(dest_trial_path, "ik.mot"))
    else:
        print(f"    - ik.mot not found for trial {trial_name}")

    # 3. Find and copy ik_sync.mot
    ik_sync_path_found = None
    ik_sync_dir = os.path.join(trial_path, "OpenSim", "IK", "shiftedIK")
    if os.path.isdir(ik_sync_dir):
        ik_sync_paths = glob.glob(os.path.join(ik_sync_dir, "*_sync.mot"))
        if ik_sync_paths:
            ik_sync_path_found = ik_sync_paths[0]

    if ik_sync_path_found:
        shutil.copy(ik_sync_path_found, os.path.join(dest_trial_path, "ik_sync.mot"))
    else:
        print(f"    - ik_sync.mot not found for trial {trial_name}")

    # 4. Find and copy lag.txt
    lag_path_found = None
    lag_dir = os.path.join(trial_path, "OpenSim", "IK", "shiftedIK")
    if os.path.isdir(lag_dir):
        lag_files = glob.glob(os.path.join(lag_dir, "lag_correlation*.txt"))
        if lag_files:
            lag_path_found = lag_files[0]

    if lag_path_found:
        shutil.copy(lag_path_found, os.path.join(dest_trial_path, "lag.txt"))
    else:
        print(f"    - lag.txt not found for trial {trial_name}")


def main():
    """
    Main function to organize the results.
    """
    if os.path.exists(DEST_DIR):
        print(f"Destination directory '{DEST_DIR}' already exists. Removing it.")
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)
    print(f"Created destination directory: '{DEST_DIR}'")

    SOURCE_DIRS = [
        "output/case_001_walking",
        "output/case_001_STS",
        "output/case_001_squats",
    ]

    for case_dir in SOURCE_DIRS:
        if not os.path.isdir(case_dir):
            print(f"Source case directory '{case_dir}' not found. Skipping.")
            continue

        print(f"Processing case directory: {case_dir}")
        for subject in sorted(os.listdir(case_dir)):
            subject_path = os.path.join(case_dir, subject)
            if not os.path.isdir(subject_path) or not subject.startswith("subject"):
                continue

            print(f"  Processing subject: {subject}")
            dest_subject_path = os.path.join(DEST_DIR, subject)

            for session in sorted(os.listdir(subject_path)):
                session_path = os.path.join(subject_path, session)
                if not os.path.isdir(session_path):
                    continue

                for cam in sorted(os.listdir(session_path)):
                    cam_path = os.path.join(session_path, cam)
                    if not os.path.isdir(cam_path):
                        continue

                    for trial in sorted(os.listdir(cam_path)):
                        trial_path_base = os.path.join(cam_path, trial)
                        if not os.path.isdir(trial_path_base):
                            continue

                        processing_path = find_trial_processing_path(trial_path_base)

                        if processing_path:
                            print(f"    - Found trial: {trial}")
                            if not os.path.exists(dest_subject_path):
                                os.makedirs(dest_subject_path)
                            dest_trial_path = os.path.join(dest_subject_path, trial)
                            os.makedirs(dest_trial_path, exist_ok=True)
                            find_and_copy_files(processing_path, dest_trial_path)


if __name__ == "__main__":
    main()
