#!/usr/bin/env python3
"""Validation helpers for the merge step (step 16).

`validate_data` checks that record/label/metabolite totals match the expected
fixture counts; `validate_json` defers to the deeper schema validator in
`validate_schema.py`.
"""

import logging

from pipeline.merge.validate_schema import validate_all_data_json

EXPECTED_TOTAL_RECORDS = 5168
EXPECTED_NUM_MANUAL_MASKS = 2153
EXPECTED_NUM_METABOLITES = 4154
EXPECTED_NUM_LABELS = 301


def validate_data(stats: dict) -> bool:
    """Sanity-check counts in the stats dict against expected fixtures."""
    logging.info("Validating data before writing...")
    try:
        assert stats["num_records"] == EXPECTED_TOTAL_RECORDS
        assert stats["num_img_paths"] == EXPECTED_TOTAL_RECORDS
        assert stats["num_mask_paths"] == EXPECTED_TOTAL_RECORDS
        assert stats["num_overlay_paths"] == EXPECTED_TOTAL_RECORDS
        assert stats["num_manual_masks"] == EXPECTED_NUM_MANUAL_MASKS
        assert stats["num_records"] == stats["num_labels"] + stats["num_no_labels"]
        assert stats["num_records"] == stats["num_metabolites"] + stats["num_no_metabolite"]
        assert stats["num_metabolites"] == EXPECTED_NUM_METABOLITES
        assert stats["num_records"] == stats["num_survey"] + stats["num_no_survey"]
        assert stats["total_votes"] == stats["num_acceptable_votes"] + stats["num_not_acceptable_votes"]
        assert stats["num_survey"] == stats["num_majority"] + stats["num_no_majority"]
        logging.info("Data validation passed")
        return True
    except AssertionError as e:
        logging.exception(f"Data validation failed with exception: {e}")
        return False


def validate_json(records: dict):
    """Run the deeper schema validator on the in-memory records dict."""
    logging.info("Validating schema of records before writing...")
    valid = False
    validation_results = {"valid": False, "errors": [], "warnings": [], "stats": {}}
    try:
        validation_results = validate_all_data_json(data=records, strict=True)
        if validation_results["valid"]:
            valid = True
            logging.info("Schema validation passed")
        else:
            error_count = len(validation_results["errors"])
            warning_count = len(validation_results["warnings"])
            logging.warning(f"Schema validation found {error_count} errors and {warning_count} warnings")
            for error in validation_results["errors"][:5]:
                logging.warning(f"  - {error}")
            if error_count > 5:
                logging.warning(f"  ... and {error_count - 5} more errors")
    except Exception as e:
        logging.exception(f"Schema validation failed with exception: {e}")
    return valid, validation_results["stats"]
