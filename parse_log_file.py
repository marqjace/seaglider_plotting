# Parse Seaglider logfiles and output parameters as a dictionary.

def parse_log_file(filepath):
    """
    Parses a single Seaglider logfile.
    
    Parameters:
        filepath (str): Path to the directory containing log files.
        
    Returns:
        parameters (dict): A dictionary containing parsed parameters from the logfile.

    """
    parameters = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('$'):
                parts = line[1:].split(',')  # Remove $ and split by comma
                key = parts[0]
                raw_values = parts[1:]

                values = []
                for val in raw_values:
                    if '.' in val or 'e' in val.lower():
                        try:
                            values.append(float(val))
                        except ValueError:
                            continue  # skip malformed value
                    else:
                        try:
                            values.append(int(val))
                        except ValueError:
                            continue  # skip malformed value

                # Store a single value as a scalar, or multiple as a list
                if values:
                    parameters[key] = values[0] if len(values) == 1 else values
    return parameters

# Example usage
# log_parameters = parse_log_file(r'C:\Users\marqjace\TH_line\deployments\mar_2025\transect4\logfiles\p2660402.log')
# print(log_parameters.keys())

