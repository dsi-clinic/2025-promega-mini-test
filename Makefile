.PHONY: data clean help

# Generate the master data file with plate identifiers preserved
data:
	PYTHONPATH=. conda run -p /net/projects2/promega python file_utils/merge/merge_all_data.py
# image mapper
data-image-mapper:
	PYTHONPATH=. conda run -p /net/projects2/promega python file_utils/images/image_mapper_main.py
	
# train the image classifier model
train:
	PYTHONPATH=. conda run -p /net/projects2/promega python analysis/images/classifier/train_model_accuracy.py

# Clean generated data files
clean:
	rm -f all_data.json all_data_old.json

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make data    - Generate all_data.json with preserved plate identifiers"
	@echo "  make clean   - Remove generated data files"
	@echo "  make help    - Show this help message"