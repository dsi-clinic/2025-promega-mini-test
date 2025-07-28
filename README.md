Environment changes:
- Create a conda yaml file we all agree with.
- Jiawei has a requirements.txt that needs to be incorporated
- Use Python 3.11. MMSeg is going to be a nightmare, ignore for now.
- Rename the master_mapper and mapper_main to something a bit more obvious.


Dataset changes:
- Update the directory structure to the following structure. This should handle the data processing
- file_utils/[data type]
- file_utils/survey
- file_utils/metabolite
- file_utils/images
- file_utils/merge (create the master json)
- The master json (all_data.json) file should be stored in the root directory (but also excluded from the repo)

How do we run Analysis:
- We use our default environment using conda
- We run them from the root.


Where are we going to:

- Specific analysis should have an indicator of what environement (conda) is required to run it. 
	- Should be a default one that we use 
- All analysis should be in the analysis folder
- All environment / configuration information should be in the top level.
- No hard coded paths inside any directories.
- Pull notebooks out of non-obvious subdirectories (they should generally be in analysis directory).
- Mentally we now have enough structure to have our data kinda set so that /analysis is the only messy directory.
- Eveything gets run from the top level: "python analysis/mmseg/run_mm.py".
- all_data.json (everything is keyed off of this. No looking at original data outside of the images)
- Analysis/[analysis name]: All analaysis should be in directories like this and should all use the all_data.json file as input


- /net/projects2/promega/[some subdirectory] <- mirror what is on cloudflare. Move code toward this eventually. Need to have two commands, one which mirrors up and one which mirrors down



