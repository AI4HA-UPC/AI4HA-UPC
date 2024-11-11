import csv
import os


PATH = '/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_512_v2'
# Define the CSV file containing the training images
csv_file = "train_combined.csv"
PATH_CSV = os.path.join(PATH, csv_file) 

# Define the number of parts to split the dataset into
n = 3

# Open the input CSV file for reading
with open(PATH_CSV, 'r') as input_file:
    # Create a CSV reader
    csv_reader = csv.reader(input_file)
    
    # Read the header (assuming the first row is the header)
    header = next(csv_reader)

    # Calculate the number of lines per part
    num_lines = sum(1 for _ in csv_reader)
    lines_per_part = num_lines // n

    # Reset the reader to the beginning of the file
    input_file.seek(0)
    next(csv_reader)  # Skip the header again

    part_index = 0
    current_part_lines = 0

    # Create the first output file
    output_file_path = os.path.join(PATH, f"train_split_{part_index + 1}.csv")
    output_file = open(output_file_path, 'w', newline='')
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(header)

    # Loop through the rows
    for row in csv_reader:
        if current_part_lines >= lines_per_part:
            # Close the current output file
            output_file.close()
            part_index += 1

            if part_index >= n:
                break

            # Create a new output file
            output_file_path = os.path.join(PATH, f"train_split_{part_index + 1}.csv")
            output_file = open(output_file_path, 'w', newline='')
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(header)

            current_part_lines = 0

        csv_writer.writerow(row)
        current_part_lines += 1

    # Close the last output file
    output_file.close()

print(f"Split the dataset into {n} parts.")
