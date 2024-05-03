import argparse

def parse_arguments(ignored_params):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("args", nargs='+', help="List of arguments in the format [+]key=value")
    
    args = parser.parse_args()
    parsed_args = {}
    for arg in args.args:
        key_value = arg.split("=")
        if len(key_value) != 2:
            raise ValueError("Invalid argument format: {}".format(arg))
        key = key_value[0]
        value = key_value[1]
        if key.startswith("+"):
            key = key[1:]
        if key not in ignored_params:
            parsed_args[key] = value
    return parsed_args

if __name__ == "__main__":
    # Specify the ignored parameters here
    ignored_params = ["headless"]
    
    parsed_args = parse_arguments(ignored_params)
    print("Parsed arguments:")
    for key, value in parsed_args.items():
        print(f"{key}: {value}")