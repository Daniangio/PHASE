import sys
from typing import Any, Dict
import yaml


def load_config(config_file: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    print(f"Loading configuration from {config_file}...")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            print("Warning: Config file is empty.")
            return {}
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file}", file=sys.stderr)
        sys.exit(1) # Exit if config file is missing
    except Exception as e:
        print(f"Error loading YAML configuration: {e}", file=sys.stderr)
        sys.exit(1) # Exit on YAML parsing error