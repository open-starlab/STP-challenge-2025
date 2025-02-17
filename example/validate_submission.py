import os
import glob
import argparse
import pandas as pd


PREDICT_FRAMES = 30

parser = argparse.ArgumentParser()
parser.add_argument('--submit', type=str, required=True, help='Path to the folder containing submission tracking files.')
parser.add_argument('--input', type=str, required=True, help='Path to the folder containing input tracking files.')
args = parser.parse_args()


def get_tracking_files(folder):
    """
    Retrieve a sorted list of tracking file paths from the specified folder.

    Parameters:
        folder (str): Path to the folder containing tracking files.

    Returns:
        list: List of tracking file paths.
    """
    tracking_files = sorted(glob.glob(os.path.join(folder, '*.tracking.csv')))
    return tracking_files


def validate_submission_files(submit_files, input_files):
    """
    Validate the submitted file names against the ground truth file names.

    Parameters:
        submit_files (list): List of submission file paths.
        input_files (list): List of input file paths.

    Returns:
        bool: True if submission is valid, False otherwise.
    """

    submit_filenames = [os.path.basename(f) for f in submit_files]
    input_filenames = [os.path.basename(f) for f in input_files]

    # Check if the filenames are the same
    if set(submit_filenames) != set(input_filenames):
        print("The file names in the submission and input do not match.")
        files_in_submission_not_in_input = set(submit_filenames) - set(input_filenames)
        files_in_input_not_in_submission = set(input_filenames) - set(submit_filenames)

        if files_in_submission_not_in_input:
            print("Files in submission but not in input:", files_in_submission_not_in_input)
        else:
            print("Files in submission but not in input: None")

        if files_in_input_not_in_submission:
            print("Files in input but not in submission:", files_in_input_not_in_submission)
        else:
            print("Files in input but not in submission: None")

        return False

    return True


def validate_submission_data(submit_files, input_files):
    """
    Validate the submission data.

    Parameters:
        submit_files (list): List of submission file paths.
        input_files (list): List of input file paths.

    Returns:
        bool: True if submission is valid, False otherwise.
    """

    # Check if the submission dataframe has the expected columns
    expected_columns = [f'l{i}_x' for i in range(1, 12)] + [f'l{i}_y' for i in range(1, 12)] + ['b_x', 'b_y']

    # print("expected_columns: ", expected_columns)
    for submit_file, input_file in zip(submit_files, input_files):
        submit_df = pd.read_csv(submit_file, index_col='#')
        input_df = pd.read_csv(input_file, index_col='#')

        missing_columns = [col for col in expected_columns if col not in submit_df.columns]
        if missing_columns:
            print(f"Some columns are missing in the submitted data in {submit_file}: {missing_columns}")
            return False

        # Check if the submit_df has the expected rows from input_df
        last_index = input_df.index[-1]
        expected_indices = range(last_index + 1, last_index + 31)
        if not all(idx in submit_df.index for idx in expected_indices):
            print(f"The submitted data in {submit_file} does not contain the expected rows from {last_index + 1} to {last_index + 30}.")
            return False

    return True


def main():
    submit_folder = args.submit
    input_folder = args.input

    print("submit_folder: ", submit_folder)
    print("input_folder: ", input_folder)

    submit_files = get_tracking_files(submit_folder)
    input_files = get_tracking_files(input_folder)

    if not validate_submission_files(submit_files, input_files):
        return

    if not validate_submission_data(submit_files, input_files):
        return

    print("Submission is valid.")


if __name__ == '__main__':
    main()
