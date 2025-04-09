#!/bin/bash
set -e
# Get the current conda environment name
CURRENT_ENV=$CONDA_DEFAULT_ENV

# Check if the environment is named mmseg_env
# if [ "$CURRENT_ENV" != "mmseg_env" ]; then
#     echo "Error: Current conda environment is '$CURRENT_ENV', not 'mmseg_env'"
#     echo "Please activate the correct environment with: conda activate mmseg_env"
#     echo "To create the env type: conda env create -f mmseg_environment.yml"
#     echo "to activate the env type: source activate mmseg_env"
#     exit 1
# fi

# Get the hostname
HOST=$(hostname)

# Check if hostname begins with "fe"
if [[ "$HOST" == fe* ]]; then
    echo "Error: Do not run on login box"
    echo "Current host: $HOST"
    echo "To request a session type: srun -p general --gres=gpu:1 -t 120:00 --mem 64G --pty /bin/bash"
    exit 1
fi

echo "Running on computational node: $HOST"
echo "Running in correct environment: mmseg_env"

# Source environment variables
source .env

# Print environment variables for debugging
echo "BASE_PATH: $BASE_PATH"
echo "JSON_MAPPING_PATH: $JSON_MAPPING_PATH"

# Export all environment variables for config parsing
set -a  # automatically export all variables
source .env
set +a  # stop automatically exporting

# Run the training with the dataset path from environment variable
python train.py segformer_mitb0.py \
    --work-dir ${PLOTS_FOLDER}/segformer_masks \
    --cfg-options "train_dataloader.dataset.json_mapping_path=$JSON_MAPPING_PATH" \
                 "val_dataloader.dataset.json_mapping_path=$JSON_MAPPING_PATH"