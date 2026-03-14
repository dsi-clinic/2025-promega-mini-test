# File Utilities

Data mapping and merging utilities that produce `all_data.json` from raw data sources on the cluster.

## Directory Structure

```
file_utils/
├── README.md                      This file
├── __init__.py                    Package init
├── common/                        Shared helpers
│   ├── __init__.py
│   └── organoid_patterns.py       Organoid ID normalization and pattern matching
├── images/                        Raw image mapping pipeline
│   ├── __init__.py
│   ├── image_mapper_main.py       Entry point for building the raw image mapping JSON
│   └── scripts/
│       ├── __init__.py
│       ├── image_mapper.py        Z-stack file discovery, image classification (Regular/Split/Stitched)
│       └── mapping_system.py      Orchestrates the full image mapping workflow
├── metabolites/                   Metabolite data extraction
│   └── metabolite_mapper.py       Reads metabolite Excel, writes metabolite_map.json
├── surveys/                       Survey label aggregation
│   └── surveys_mapper.py          Parses Organoid/Image Classification Excel forms, writes aggregated JSON
└── merge/                         Final merge step
    └── merge_all_data.py          Combines image mapping, processed images, surveys, metabolites, and manual masks into all_data.json
```

## Usage

The standard pipeline runs in order:

```bash
make data
```

Or individually:

```bash
python file_utils/images/image_mapper_main.py
python file_utils/metabolites/metabolite_mapper.py
python file_utils/surveys/surveys_mapper.py
python file_utils/merge/merge_all_data.py
```

All scripts read paths from `config.py` / `.env`. See the [root README](../README.md) for environment setup.
