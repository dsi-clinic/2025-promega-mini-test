#!/usr/bin/env python3
"""
Centralized regex patterns for organoid key normalization.

This module provides consistent patterns and utilities for parsing and normalizing
organoid identifiers across the codebase, eliminating duplicate regex patterns.
"""

import re
from typing import NamedTuple, Optional, Dict, Pattern


class OrganoidPatterns:
    """Centralized regex patterns for organoid key normalization"""
    
    # Batch patterns
    BATCH_STRICT = re.compile(r'^BA(\d+)$', re.IGNORECASE)
    BATCH_FLEXIBLE = re.compile(r'\b(?:BA|Batch)\s*(\d+)\b', re.IGNORECASE)
    BATCH_TOKEN = re.compile(r'^BA\d+$', re.IGNORECASE)
    
    # Day patterns  
    DAY_STRICT = re.compile(r'^DY(\d+)$', re.IGNORECASE)
    DAY_FLEXIBLE = re.compile(r'\b(?:Dy|Day)\s*(\d+)\b', re.IGNORECASE)
    DAY_EXTRACT = re.compile(r'[Dd][Yy]?(\d+)', re.IGNORECASE)
    DAY_EXTRACT_WORD_BOUNDARY = re.compile(r'\bDy(\d{1,2})\b', re.IGNORECASE)
    DAY_TOKEN = re.compile(r'^DY\d+$', re.IGNORECASE)
    
    # Plate patterns
    PLATE_PATTERN = re.compile(r'\b(96_[12]|PT1)\b', re.IGNORECASE)
    PLATE_REMOVE = re.compile(r'96_[12]', re.IGNORECASE)
    PLATE_TOKEN = re.compile(r'^(96_[12]|PT1)$', re.IGNORECASE)
    
    # Well ID patterns
    WELL_STRICT = re.compile(r'(?<!BA)\b([A-H])(\d{1,2})\b', re.IGNORECASE)
    WELL_FLEXIBLE = re.compile(r'^([A-Ha-h])\s*([1-9]|1[0-2])$')
    
    # Cleaning patterns
    REMOVE_BRACKETS = re.compile(r'\[.*?\]')
    REMOVE_PARENS = re.compile(r'\(.*?\)')
    KEEP_ALPHANUMERIC = re.compile(r'[^A-Za-z0-9\s_]')
    NORMALIZE_SPACES = re.compile(r'\s+')
    NORMALIZE_SEPARATORS = re.compile(r'[_\-]+')
    
    # File/path patterns
    RESOLUTION_EXTRACT = re.compile(r'processed_dataset_(\d+x\d+)')
    BATCH_DAY_PATH = re.compile(r'/batch(\d+)/day(\d+)/', re.IGNORECASE)
    Z_LEVEL = re.compile(r' Z(\d+)', re.IGNORECASE)
    
    # Image file patterns
    PARTIAL_IMAGE = re.compile(r'\(\d+\s+of\s+\d+\)')
    DUPLICATE_IMAGE = re.compile(r'\((\d+)\)')
    HASH_PERCENT = re.compile(r'\(#\)%')
    
    # Path matching patterns
    BA_SUBSTITUTE = re.compile(r'\bBA3\b', re.IGNORECASE)
    
    STITCHED      = re.compile(r"\(stitched\)", re.IGNORECASE)


    # split markers AFTER the split day
    SPLIT_PAREN  = re.compile(r"\(\s*(\d+)\s*\)\s*%", re.IGNORECASE)   # e.g., C1(1)% Z0.tif
    SPLIT_HYPHEN = re.compile(r"-\s*(\d)\s*-%", re.IGNORECASE)         # e.g., D12-2-% ...

class OrganoidKey(NamedTuple):
    """Structured representation of an organoid key"""
    batch: str
    plate: Optional[str]
    day: str  
    well: str


class OrganoidNormalizer:
    """Handles organoid key normalization using centralized patterns"""
    
    @staticmethod
    def clean_string(text: str) -> str:
        """Standard string cleaning for organoid IDs"""
        text = OrganoidPatterns.REMOVE_BRACKETS.sub("", text)
        text = OrganoidPatterns.REMOVE_PARENS.sub("", text) 
        text = OrganoidPatterns.KEEP_ALPHANUMERIC.sub(" ", text)
        text = OrganoidPatterns.NORMALIZE_SPACES.sub(" ", text).strip()
        return text
        
    @staticmethod
    def normalize_separators(text: str) -> str:
        """Convert underscores and dashes to spaces"""
        return OrganoidPatterns.NORMALIZE_SEPARATORS.sub(' ', text).strip()
        
    @staticmethod
    def extract_batch(text: str) -> Optional[str]:
        """Extract batch identifier (BA1, BA2, etc.)"""
        match = OrganoidPatterns.BATCH_FLEXIBLE.search(text)
        return f"BA{match.group(1)}" if match else None
        
    @staticmethod
    def extract_day(text: str) -> Optional[str]:
        """Extract day identifier (Dy03, Dy30, etc.)"""
        match = OrganoidPatterns.DAY_EXTRACT.search(text)
        return f"Dy{int(match.group(1)):02d}" if match else None
        
    @staticmethod
    def extract_day_number(text: str) -> Optional[int]:
        """Extract day number as integer"""
        match = OrganoidPatterns.DAY_EXTRACT.search(text)
        return int(match.group(1)) if match else None
        
    @staticmethod 
    def extract_well(text: str) -> Optional[str]:
        """Extract well identifier (A1, H12, etc.)"""
        match = OrganoidPatterns.WELL_STRICT.search(text)
        return f"{match.group(1).upper()}{match.group(2)}" if match else None
        
    @staticmethod
    def extract_plate(text: str) -> Optional[str]:
        """Extract plate identifier (96_1, 96_2, PT1)"""
        match = OrganoidPatterns.PLATE_PATTERN.search(text)
        return match.group(1) if match else None
        
    @staticmethod
    def extract_resolution(text: str) -> Optional[str]:
        """Extract resolution from path (e.g., '256x192')"""
        match = OrganoidPatterns.RESOLUTION_EXTRACT.search(text)
        return match.group(1) if match else None
        
    @staticmethod
    def extract_z_level(text: str) -> int:
        """Extract Z level from filename, returns -1 if not found"""
        match = OrganoidPatterns.Z_LEVEL.search(text)
        return int(match.group(1)) if match else -1
        
    @staticmethod
    def parse_organoid_key(text: str) -> OrganoidKey:
        """Parse text into structured organoid key components"""
        cleaned = OrganoidNormalizer.clean_string(text)
        
        batch = OrganoidNormalizer.extract_batch(cleaned)
        day = OrganoidNormalizer.extract_day(cleaned)
        well = OrganoidNormalizer.extract_well(cleaned)
        plate = OrganoidNormalizer.extract_plate(cleaned)
        
        if not all([batch, day, well]):
            raise ValueError(f"Could not extract required components (batch={batch}, day={day}, well={well}) from: {text}")
            
        return OrganoidKey(batch=batch, plate=plate, day=day, well=well)
        
    @staticmethod
    def normalize_key(text: str) -> str:
        """Convert raw text to normalized organoid key format (BA1 96_1 Dy30 A1)"""
        key = OrganoidNormalizer.parse_organoid_key(text)
        
        # Build normalized key
        ba_part = f"{key.batch} {key.plate}" if key.plate else key.batch
        return f"{ba_part} {key.day} {key.well}"

    @staticmethod
    def extract_split_info(raw_name: str) -> dict:
        f = raw_name.lower()
        info = {
            "is_split": False,
            "pre_split": False,   # kept for backward compatibility, always False now
            "split_index": None,
            "stitched": False,
            "partial": False
        }

        m = OrganoidPatterns.SPLIT_PAREN.search(f) or OrganoidPatterns.SPLIT_HYPHEN.search(f)
        if m:
            info["is_split"] = True
            info["split_index"] = int(m.group(1))

        if OrganoidPatterns.STITCHED.search(f):
            info["stitched"] = True
        if OrganoidPatterns.PARTIAL_IMAGE.search(f):
            info["partial"] = True

        return info

class OrganoidValidation:
    """Validation utilities for organoid keys and components"""
    
    @staticmethod
    def is_valid_batch_token(text: str) -> bool:
        """Check if text is a valid batch token (BA1, BA2, etc.)"""
        return bool(OrganoidPatterns.BATCH_TOKEN.match(text))
    
    @staticmethod  
    def is_valid_day_token(text: str) -> bool:
        """Check if text is a valid day token (DY01, DY30, etc.)"""
        return bool(OrganoidPatterns.DAY_TOKEN.match(text))
        
    @staticmethod
    def is_valid_plate_token(text: str) -> bool:
        """Check if text is a valid plate token (96_1, 96_2, PT1)"""
        return bool(OrganoidPatterns.PLATE_TOKEN.match(text))
        
    @staticmethod
    def is_valid_organoid_key(key: str) -> bool:
        """Validate complete organoid key format"""
        try:
            OrganoidNormalizer.parse_organoid_key(key)
            return True
        except ValueError:
            return False


# Convenience functions for backward compatibility
def norm_key(text: str) -> str:
    """Normalize organoid key - convenience function for existing code"""
    return OrganoidNormalizer.normalize_key(text)

def day_from_key(key: str) -> Optional[int]:
    """Extract day number from key - convenience function for existing code"""
    return OrganoidNormalizer.extract_day_number(key)

def clean_id_for_json(text: str) -> str:
    """Clean ID for JSON - convenience function for existing code"""
    return OrganoidNormalizer.clean_string(text)