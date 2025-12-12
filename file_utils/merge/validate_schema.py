#!/usr/bin/env python3
"""Light schema validation for all_data.json file."""

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SchemaValidationError(Exception):
    """Custom exception for schema validation errors."""
    pass


class ValidationContext:
    """Context object to hold validation state and helper methods."""

    def __init__(self, record_id: str, errors: List[str], warnings: List[str],
                 stats: Dict[str, Any], strict: bool = False):
        self.record_id = record_id
        self.errors = errors
        self.warnings = warnings
        self.stats = stats
        self.strict = strict

    def check_type(self, obj: Any, expected_type: Union[type, Tuple[type, ...]],
                   field_path: str, required: bool = True) -> bool:
        """Check if object is of expected type."""
        if obj is None:
            if required:
                self.errors.append(f"Record {self.record_id}: {field_path} is required but is None")
            return False
        if not isinstance(obj, expected_type):
            type_name = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
            self.errors.append(f"Record {self.record_id}: {field_path} must be {type_name}, got {type(obj).__name__}")
            return False
        return True

    def check_required_fields(self, obj: Dict[str, Any], required_fields: List[str],
                            field_path: str) -> bool:
        """Check if all required fields are present in object."""
        if not isinstance(obj, dict):
            return False
        missing = [f for f in required_fields if f not in obj]
        if missing:
            self.errors.append(f"Record {self.record_id}: Missing {field_path} fields: {missing}")
            return False
        return True

    def check_nested_structure(self, parent: Dict[str, Any], field_name: str,
                              required_fields: List[str], field_path: str) -> Optional[Dict[str, Any]]:
        """Check nested structure exists and has required fields."""
        nested = parent.get(field_name, {})
        if not self.check_type(nested, dict, f"{field_path}.{field_name}"):
            return None
        if nested and not self.check_required_fields(nested, required_fields, f"{field_path}.{field_name}"):
            return None
        return nested

    def validate_path(self, path: Optional[str], field_path: str) -> None:
        """Validate file path exists (only in strict mode)."""
        if path and self.strict and not Path(path).exists():
            self.warnings.append(f"Record {self.record_id}: {field_path} does not exist: {path}")

    def track_distribution(self, key: str, value: str) -> None:
        """Track distribution of values in stats."""
        if key not in self.stats:
            self.stats[key] = {}
        self.stats[key][value] = self.stats[key].get(value, 0) + 1


def validate_path_fields(ctx: ValidationContext, processed: Dict[str, Any],
                        path_fields: List[str], base_path: str) -> None:
    """Validate multiple path fields in processed images."""
    for field in path_fields:
        path_value = processed.get(field)
        if path_value is not None:
            if not ctx.check_type(path_value, str, f"{base_path}.{field}"):
                continue
            ctx.validate_path(path_value, f"{base_path}.{field}")


def validate_day_structure(ctx: ValidationContext, day: Dict[str, Any]) -> None:
    """Validate day structure."""
    required_day_fields = ['id', 'number', 'original']
    if not ctx.check_required_fields(day, required_day_fields, 'day'):
        return

    # Validate day number
    if 'number' in day and day['number'] is not None:
        if not ctx.check_type(day['number'], (int, float), 'day.number'):
            pass  # Error already added
        elif day['number'] < 0 or day['number'] > 100:
            ctx.warnings.append(f"Record {ctx.record_id}: day.number ({day['number']}) is outside expected range [0, 100]")

    # Track day distribution
    day_id = day.get('id', 'unknown')
    ctx.track_distribution('day_distribution', day_id)


def validate_plate_structure(ctx: ValidationContext, plate: Dict[str, Any]) -> None:
    """Validate plate structure."""
    required_plate_fields = ['batch', 'well']
    ctx.check_required_fields(plate, required_plate_fields, 'plate')


def validate_metadata_structure(ctx: ValidationContext, metadata: Dict[str, Any]) -> None:
    """Validate metadata structure."""
    required_metadata_fields = ['classification', 'verification']
    if not ctx.check_required_fields(metadata, required_metadata_fields, 'metadata'):
        return

    # Validate verification structure
    verification = metadata.get('verification', {})
    if verification:
        required_verification_fields = ['classification_verification']
        if ctx.check_nested_structure(metadata, 'verification', required_verification_fields, 'metadata'):
            class_verif = verification.get('classification_verification', 'unknown')
            ctx.track_distribution('classification_verification_distribution', class_verif)


def validate_images_structure(ctx: ValidationContext, images: Dict[str, Any]) -> None:
    """Validate images structure."""
    if not ctx.check_type(images, dict, 'images'):
        return

    if images:
        ctx.stats['records_with_images'] += 1

        required_images_fields = ['processed', 'raw_images', 'preprocessed']
        ctx.check_required_fields(images, required_images_fields, 'images')

        # Validate processed structure
        processed = images.get('processed', {})
        if processed:
            required_processed_fields = ['img_path', 'mask_path', 'overlay_path']
            if ctx.check_nested_structure(images, 'processed', required_processed_fields, 'images'):
                validate_path_fields(ctx, processed, required_processed_fields, 'images.processed')

        # Validate raw_images
        raw_images = images.get('raw_images', [])
        if raw_images:
            ctx.check_type(raw_images, list, 'images.raw_images')

        # Validate preprocessed
        preprocessed = images.get('preprocessed', {})
        if preprocessed:
            ctx.check_type(preprocessed, dict, 'images.preprocessed')


def validate_metabolites_structure(ctx: ValidationContext, metabolites: Dict[str, Any]) -> None:
    """Validate metabolites structure."""
    if not metabolites:
        return

    ctx.stats['records_with_metabolites'] += 1
    if not ctx.check_type(metabolites, dict, 'metabolites'):
        return

    required_metabolites_fields = ['concentration_uM', 'is_outlier']
    expected_metabolites = ['BCAAGlo', 'GlucoseGlo', 'GlutamateGlo', 'LactateGlo', 'MalateGlo', 'PyruvateGlo']

    for metab_key, metab_data in metabolites.items():
        if not ctx.check_type(metab_data, dict, f'metabolites.{metab_key}'):
            continue

        # Check required fields
        if not ctx.check_required_fields(metab_data, required_metabolites_fields, f'metabolites.{metab_key}'):
            continue

        # Validate data types
        if 'concentration_uM' in metab_data:
            conc = metab_data['concentration_uM']
            if conc is not None:
                ctx.check_type(conc, (int, float), f'metabolites.{metab_key}.concentration_uM')

        if 'is_outlier' in metab_data:
            ctx.check_type(metab_data['is_outlier'], bool, f'metabolites.{metab_key}.is_outlier')


def validate_survey_structure(ctx: ValidationContext, survey: Dict[str, Any]) -> None:
    """Validate survey structure."""
    if not survey:
        return

    ctx.stats['records_with_survey'] += 1
    if not ctx.check_type(survey, dict, 'survey'):
        return

    required_survey_fields = ['evaluations', 'quality_scores', 'summary', 'label']
    if not ctx.check_required_fields(survey, required_survey_fields, 'survey'):
        return

    # Validate list fields
    for list_field in ['evaluations', 'quality_scores']:
        field_value = survey.get(list_field, [])
        if field_value:
            ctx.check_type(field_value, list, f'survey.{list_field}')

    # Validate summary
    summary = survey.get('summary', {})
    if summary:
        ctx.check_type(summary, dict, 'survey.summary')

    # Validate label structure
    label = survey.get('label', {})
    if label:
        required_label_fields = ['value', 'acceptance_flag']
        if ctx.check_nested_structure(survey, 'label', required_label_fields, 'survey'):
            # Validate label value
            if 'value' in label and label['value']:
                if label['value'] not in ['Acceptable', 'Not Acceptable']:
                    ctx.warnings.append(f"Record {ctx.record_id}: survey.label.value has unexpected value: {label['value']}")

            # Validate acceptance_flag
            if 'acceptance_flag' in label:
                flag = label['acceptance_flag']
                if flag is not None:
                    if not ctx.check_type(flag, int, 'survey.label.acceptance_flag', required=False):
                        pass  # Error already added
                    elif flag not in [0, 1]:
                        ctx.warnings.append(f"Record {ctx.record_id}: survey.label.acceptance_flag has unexpected value: {flag} (expected 0 or 1)")


def validate_label_structure(ctx: ValidationContext, label: Dict[str, Any]) -> None:
    # Validate label structure
    if not label:
        return
    required_label_fields = ['value', 'acceptance_flag', 'source']
    if ctx.check_nested_structure(label, 'label', required_label_fields, 'label'):
        # Validate label value
        if 'value' in label and label['value']:
            if label['value'] not in ['Acceptable', 'Not Acceptable']:
                ctx.warnings.append(f"Record {ctx.record_id}: label.value has unexpected value: {label['value']}")

        # Validate acceptance_flag
        if 'acceptance_flag' in label:
            flag = label['acceptance_flag']
            if flag is not None:
                if not ctx.check_type(flag, int, 'survey.label.acceptance_flag', required=False):
                    pass  # Error already added
                elif flag not in [0, 1]:
                    ctx.warnings.append(f"Record {ctx.record_id}: survey.label.acceptance_flag has unexpected value: {flag} (expected 0 or 1)")

        # Validate source
        if 'source' in label:
            source = label['source']
            if source is not None:
                if not ctx.check_type(source, str, 'label.source', required=False):
                    pass  # Error already added
                elif source not in ['survey.evaluations', 'preprocessed.label', None]:
                    ctx.warnings.append(f"Record {ctx.record_id}: label.source has unexpected value: {source} (expected 'image.label', 'survey.label', or 'preprocessed.label')")


def validate_record_id_format(ctx: ValidationContext, record_id: str) -> None:
    """Validate record ID format."""
    # Pattern: BA#_96_#_Dy##_A# optionally followed by _split_# (e.g., BA1_96_1_Dy03_B5 or BA4_96_1_Dy20.5_C12_split_1)
    # - [A-Z]+\d+ : Batch identifier (BA1)
    # - (_\d+)? : Optional plate number (_96)
    # - _\d+ : Additional plate identifier (_1)
    # - _Dy\d+(\.\d+)? : Day identifier (_Dy03 or _Dy20.5)
    # - _[A-Z]\d+ : Well identifier (_B5)
    # - (_split_\d+)? : Optional split identifier (_split_1, _split_2, etc.)
    pattern = r'^[A-Z]+\d+(_\d+)?_\d+_Dy\d+(\.\d+)?_[A-Z]\d+(_split_\d+)?$'
    if not re.match(pattern, record_id):
        ctx.warnings.append(f"Record {record_id}: ID format may be unexpected (expected pattern: BA#_96_#_Dy##_A# or BA#_96_#_Dy##.#_A#_split_#)")


def validate_records_dict(data: Dict[str, Any], sample_size: Optional[int] = None, strict: bool = False) -> Dict[str, Any]:
    """
    Perform light schema validation on a records dictionary (in-memory).

    Args:
        data: Dictionary of records to validate (e.g., records_clean from merge_all_data.py)
        sample_size: Number of records to sample for validation (None = validate all)
        strict: If True, fail on warnings; if False, only fail on errors

    Returns:
        Dictionary with validation results: {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'stats': Dict[str, Any]
        }
    """
    errors: List[str] = []
    warnings: List[str] = []
    stats = {
        'total_records': 0,
        'records_with_required_fields': 0,
        'records_with_images': 0,
        'records_with_metabolites': 0,
        'records_with_survey': 0,
        'day_distribution': {},
        'classification_verification_distribution': {},
    }

    # 1. Check top-level structure
    if not isinstance(data, dict):
        errors.append("Top-level structure must be a dictionary/object")
        return {'valid': False, 'errors': errors, 'warnings': warnings, 'stats': stats}

    if len(data) == 0:
        errors.append("Data is empty (no records)")
        return {'valid': False, 'errors': errors, 'warnings': warnings, 'stats': stats}

    # 3. Define required fields
    required_fields = ['id', 'day', 'cell_line', 'plate', 'metadata', 'images', 'metabolites', 'survey']

    # 4. Sample records if requested
    record_items = list(data.items())
    if sample_size and sample_size < len(record_items):
        record_items = random.sample(record_items, sample_size)
        logger.info(f"Validating sample of {sample_size} records out of {len(data)} total")

    # 5. Validate each record
    for record_id, record in record_items:
        stats['total_records'] += 1
        ctx = ValidationContext(record_id, errors, warnings, stats, strict)

        # 5a. Check required top-level fields
        missing_fields = [f for f in required_fields if f not in record]
        if missing_fields:
            errors.append(f"Record {record_id}: Missing required fields: {missing_fields}")
            continue

        stats['records_with_required_fields'] += 1

        # 5b-5g. Validate nested structures
        day = record.get('day', {})
        if day:
            validate_day_structure(ctx, day)

        plate = record.get('plate', {})
        if plate:
            validate_plate_structure(ctx, plate)

        metadata = record.get('metadata', {})
        if metadata:
            validate_metadata_structure(ctx, metadata)

        images = record.get('images', {})
        if images:
            validate_images_structure(ctx, images)

        metabolites = record.get('metabolites', {})
        if metabolites:
            validate_metabolites_structure(ctx, metabolites)

        survey = record.get('survey', {})
        if survey:
            validate_survey_structure(ctx, survey)


        label = record.get('label', {})
        if label:
            validate_label_structure(ctx, label)

        # 5h. Validate record ID format
        validate_record_id_format(ctx, record_id)

    # 6. Summary checks
    if stats['total_records'] == 0:
        errors.append("No records found in file")

    if stats['records_with_required_fields'] < stats['total_records'] * 0.95:
        warnings.append(f"Only {stats['records_with_required_fields']}/{stats['total_records']} records have all required fields (< 95%)")

    # 7. Return results
    is_valid = len(errors) == 0 and (not strict or len(warnings) == 0)

    return {
        'valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'stats': stats
    }


def validate_all_data_json(json_path: Optional[Path] = None, data: Optional[Dict[str, Any]] = None,
                           sample_size: Optional[int] = None, strict: bool = False) -> Dict[str, Any]:
    """
    Perform light schema validation on all_data.json (file or in-memory dict).

    Args:
        json_path: Path to all_data.json file (optional if data is provided)
        data: Dictionary of records to validate (optional if json_path is provided)
        sample_size: Number of records to sample for validation (None = validate all)
        strict: If True, fail on warnings; if False, only fail on errors

    Returns:
        Dictionary with validation results: {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'stats': Dict[str, Any]
        }

    Raises:
        SchemaValidationError: If neither json_path nor data is provided, or if file doesn't exist
    """
    # Load data from file if json_path is provided
    if json_path is not None:
        if not json_path.exists():
            raise SchemaValidationError(f"File does not exist: {json_path}")

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"Invalid JSON: {e}")
    elif data is None:
        raise SchemaValidationError("Either json_path or data must be provided")

    # Validate the data dictionary
    return validate_records_dict(data, sample_size=sample_size, strict=strict)


def log_validation_report(results: Dict[str, Any]) -> None:
    """Log a formatted validation report."""
    logger.info("")
    logger.info("="*60)
    logger.info("SCHEMA VALIDATION REPORT")
    logger.info("="*60)

    if results['valid']:
        logger.info("✅ VALIDATION PASSED")
    else:
        logger.error("❌ VALIDATION FAILED")

    logger.info(f"Total records checked: {results['stats']['total_records']}")
    logger.info(f"Records with required fields: {results['stats']['records_with_required_fields']}")
    logger.info(f"Records with images: {results['stats']['records_with_images']}")
    logger.info(f"Records with metabolites: {results['stats']['records_with_metabolites']}")
    logger.info(f"Records with survey: {results['stats']['records_with_survey']}")

    if results['stats']['day_distribution']:
        logger.info(f"Day distribution (top 10):")
        sorted_days = sorted(results['stats']['day_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
        for day, count in sorted_days:
            logger.info(f"  {day}: {count}")

    if results['errors']:
        logger.error(f"\n❌ ERRORS ({len(results['errors'])}):")
        for i, error in enumerate(results['errors'][:20], 1):  # Show first 20
            logger.error(f"  {i}. {error}")
        if len(results['errors']) > 20:
            logger.error(f"  ... and {len(results['errors']) - 20} more errors")

    if results['warnings']:
        logger.warning(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for i, warning in enumerate(results['warnings'][:20], 1):  # Show first 20
            logger.warning(f"  {i}. {warning}")
        if len(results['warnings']) > 20:
            logger.warning(f"  ... and {len(results['warnings']) - 20} more warnings")

    logger.info("="*60)
    logger.info("")


def main():
    """CLI entry point for schema validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate all_data.json schema")
    parser.add_argument('json_file', type=Path, help='Path to all_data.json file')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of records to sample for validation (default: validate all)')
    parser.add_argument('--strict', action='store_true',
                       help='Treat warnings as errors')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress validation report output (only show errors)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)

    try:
        results = validate_all_data_json(json_path=args.json_file, sample_size=args.sample, strict=args.strict)

        if not args.quiet:
            log_validation_report(results)
        elif not results['valid']:
            # In quiet mode, still log errors if validation failed
            if results['errors']:
                logger.error(f"Validation failed with {len(results['errors'])} errors")
                for error in results['errors'][:10]:
                    logger.error(f"  - {error}")
                if len(results['errors']) > 10:
                    logger.error(f"  ... and {len(results['errors']) - 10} more errors")

        # Exit with error code if validation failed
        exit(0 if results['valid'] else 1)

    except SchemaValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        exit(1)
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
