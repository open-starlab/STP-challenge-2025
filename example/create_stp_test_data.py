import os
import sys
import gzip
import pandas as pd


def process_dir(input_dir, output_dir):
    """
    Process the files in the directory.
    Args:
        input_dir (str): Path to the directory.
        output_dir (str): Path to the output directory.
    """
    if not os.path.isdir(input_dir):
        print(f"{input_dir} is not a directory.")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".tracking.csv"):
            tracking_csv = os.path.join(input_dir, filename)
            # Get the filename without the extension
            filename_without_ext = filename[:-len(".tracking.csv")]
            print(f"Processing: {filename_without_ext}", file=sys.stderr)

            # Get the target game cycle to split the data
            split_cycle = get_stp_test_cycle(tracking_csv)
            if split_cycle != -1:
                print(f">>> Found. split at cycle={split_cycle}", file=sys.stderr)
                #print(f"{filename_without_ext},{split_cycle}")
                split_tracking_csv(input_dir, output_dir, filename_without_ext, split_cycle)
                split_event_csv(input_dir, output_dir, filename_without_ext, split_cycle)
                split_rcg(input_dir, output_dir, filename_without_ext, split_cycle)
                split_rcl(input_dir, output_dir, filename_without_ext, split_cycle)


def get_stp_test_cycle(tracking_csv):
    """
    Process the tracking file.
    Try to get the target game cycle for the Soccer Trajectory Prediction task.
    In 2025, the target game cycle is 30 cycles before the left team's goal scene.
    However, play_on must be continuous from the goal scene until 50 cycles before.
    Args:
        tracking_csv (str): Path to the tracking csv file.
    Returns:
        int: The number of game cycles 30 cycles prior to the goal scene that meets the criteria. If not found, return -1.
    """
    # Read the tracking csv file
    df = pd.read_csv(tracking_csv)

    # Get the frames where the playmode is goal_l
    goal_left = df[df["playmode"] == "goal_l"]
    if goal_left.empty:
        print("... Left goal not found.", file=sys.stderr)
        return -1

    # Remove consecutive rows
    goal_left = goal_left.loc[goal_left.index.to_series().diff().ne(1).cumsum().duplicated(keep='first') == False]

    goal_left_indices = goal_left.index.tolist()
    print(f'... left goal indices = {goal_left_indices}', file=sys.stderr)
    print(f'... left goal cycles = {goal_left["cycle"].tolist()}', file=sys.stderr)

    # Check playmode from 50 cycles before the goal scene
    for goal_left_index in goal_left_indices:
        result_cycle = get_split_cycle_before_goal(df, goal_left_index)
        if result_cycle != -1:
            return result_cycle

    return -1


def get_split_cycle_before_goal(df, goal_index):
    """
    Check if the goal scene meets the criteria.
    Args:
        df (pd.DataFrame): The tracking data.
        goal_index (int): The index of the goal scene.
    Returns:
        int: The number of game cycles 30 cycles before the goal scene that meets the criteria. If not found, return -1.
    """
    # Check whether the value of 'playmode' column is 'play_on'
    # for 50 consecutive frames before the goal scene
    goal_left_cycle = df.loc[goal_index, "cycle"]
    print(f'... check the goal scene at {goal_left_cycle} (index={goal_index})', file=sys.stderr)
    for i in range(goal_index - 50, goal_index):
        if df.loc[i, "playmode"] != "play_on":
            return -1

    return goal_left_cycle - 30


def split_csv(input_dir, output_dir, filename_without_ext, ext, column_index, split_cycle):
    """
    Process the tracking csv file.
    Args:
        dir_path (str): Path to the directory.
        filename_without_ext (str): The filename without the extension.
        split_cycle (int): The target game cycle.
    """
    input_file = os.path.join(input_dir, f"{filename_without_ext}.{ext}.csv")
    output_file = os.path.join(output_dir, f"{filename_without_ext}.{ext}.csv")

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        header = infile.readline()
        outfile.write(header)
 
        for line in infile:
            cycle = int(line.split(',')[column_index]) 
            if cycle > split_cycle:
                break
            outfile.write(line)


def split_tracking_csv(input_dir, output_dir, filename_without_ext, split_cycle):
    """
    Process the tracking csv file.
    Create a new tracking csv file that contains the data up to the target cycle.
    Args:
        dir_path (str): Path to the directory.
        filename_without_ext (str): The filename without the extension.
        split_cycle (int): The target game cycle.
    """
    split_csv(input_dir, output_dir, filename_without_ext, "tracking", 1, split_cycle)


def split_event_csv(input_dir, output_dir, filename_without_ext, split_cycle):
    """
    Process the event csv file.
    Create a new event csv file that contains the data up to the target cycle.
    Args:
        dir_path (str): Path to the directory.
        filename_without_ext (str): The filename without the extension.
        split_cycle (int): The target game cycle.
    """
    split_csv(input_dir, output_dir, filename_without_ext, "event", 9, split_cycle)


def split_rcg(input_dir, output_dir, filename_without_ext, split_cycle):
    """
    Process the rcg file.
    Create a new rcg file that contains the data up to the target cycle.
    Args:
        dir_path (str): Path to the directory.
        filename_without_ext (str): The filename without the extension.
        split_cycle (int): The target game cycle.
    """
    input_file = os.path.join(input_dir, f"{filename_without_ext}.rcg.gz")
    output_file = os.path.join(output_dir, f"{filename_without_ext}.rcg.gz")

    with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
        finished = False
        for line in infile:
            parts = line.split()
            if len(parts) > 1 and parts[1].isdigit():
                cycle = int(parts[1])
                if cycle > split_cycle:
                    finished = True
                    break
            if finished:
                break
            else:
                outfile.write(line)


def split_rcl(input_dir, output_dir, filename_without_ext, split_cycle):
    """
    Process the rcl file.
    Create a new rcl file that contains the data up to the target cycle.
    Args:
        dir_path (str): Path to the directory.
        filename_without_ext (str): The filename without the extension.
        split_cycle (int): The target game cycle.
    """
    input_file = os.path.join(input_dir, f"{filename_without_ext}.rcl.gz")
    output_file = os.path.join(output_dir, f"{filename_without_ext}.rcl.gz")

    with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
        finished = False
        for line in infile:
            parts = line.split(',')
            if len(parts) > 0 and parts[0].isdigit():
                cycle = int(parts[0])
                if cycle > split_cycle:
                    finished = True
                    break
            if finished:
                break
            else:
                outfile.write(line)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_stp_test_data.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"{input_dir} is not a directory.")
        sys.exit(1)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exist.")
        sys.exit(1)

    process_dir(input_dir, output_dir)
