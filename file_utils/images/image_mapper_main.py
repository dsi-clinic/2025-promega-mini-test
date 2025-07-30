"""This module provides the main entry point for the mapping system."""

import logging
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from file_utils.images.scripts.mapping_system import MappingSystem


def main() -> dict:
    """Main execution function for the entire system.

    Initializes the MappingSystem, generates the mapping, and logs the outcome.
    """
    try:
        # Initialize and run mapping system
        mapping_system = MappingSystem()
        mapping = mapping_system.generate_key_mapping()
        logging.info("Mapping generation completed successfully")
        return mapping
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
