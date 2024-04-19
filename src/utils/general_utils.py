from datetime import datetime

def print_with_time(message):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_time}] {message}")
    
def print_with_line(message):
    print("-"*100)
    print_with_time(message)
    
def print_with_star(message):
    print("*"*100)
    print_with_time(message)

def read_txt_to_list(input_file):
    # Initialize an empty list to store the item paths
    item_paths = []

    # Open the file in read mode
    with open(input_file, "r") as f:
        # Read each line in the file
        for line in f:
            # Remove trailing whitespace and newline characters
            line = line.strip()
            # Append the cleaned line to the list
            item_paths.append(line)
    return item_paths