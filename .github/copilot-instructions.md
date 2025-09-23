# LUAD Segmentation Project - AI Assistant Instructions

## Project Overview
This project focuses on semantic segmentation for lung adenocarcinoma (LUAD) histopathology images. It uses a modern PyTorch-based deep learning approach with PyTorch Lightning for training organization.

## Architecture

### Core Components
- **Data Pipeline**: Processes raw histopathology images into tiles for training and validation
- **Model Architecture**: Uses encoder-decoder architecture (PatSeg) with pre-trained foundation models
- **Training Framework**: PyTorch Lightning modules manage the training loop and validation

### Key Files and Directories
- `luadseg/data/`: Contains dataset and dataloader implementations
  - `pl_data.py`: PyTorch Lightning DataModule for managing data splits
  - `preprocess_validation.py`: CLI tool for preprocessing validation images into tiles
- `luadseg/models/`: Contains model implementations
  - `foundation_models.py`: Loads pre-trained models from HuggingFace
  - `segmentation/patseg.py`: Main segmentation model architecture
  - `pl_wrapper.py`: PyTorch Lightning module wrapping the model
- `luadseg/training/`: Contains training scripts
  - `train.py`: CLI tool for model training

## Development Workflow

### Environment Setup
```bash
# Create conda environment from environment.yaml
conda env create -f environment.yaml
conda activate torchpl
```

### Data Processing
```bash
# Preprocess validation images into tiles
python -m luadseg.data.preprocess_validation --images-dir <path> --masks-dir <path> --split-csv <path> --output-dir <path>
```

### Training
```bash
# Train the model
python -m luadseg.training.train --root-data-dir data/processed/ANORAK_not_resized --batch-size 4
```

### Code Style
```bash
# Format code
make format

# Check code quality
make lint
```

## Project Conventions

### Data Organization
- Input images and masks are stored in separate directories
- Training/validation/test splits are defined in CSV files
- Pre-processed tiles are stored in subdirectories based on tile size

### Model Structure
- Encoders are wrapped in `BaseEncoder` subclasses for consistent interface
- PatSeg model requires 4 layers for skip connections from encoder
- PyTorch Lightning is used for training organization

### Debugging Tips
- Use VS Code debugger with the provided launch.json configuration
- TensorBoard logs are stored in `logs/tb_logs/`
- Model checkpoints use the pattern `patseg-{epoch:02d}-{val_loss:.2f}`

### Dependencies
- PyTorch and PyTorch Lightning for model implementation and training
- Albumentations for data augmentation
- HuggingFace Hub for pretrained model loading
- Click for CLI interfaces

## Common Patterns
- CLI tools use Click for argument parsing
- Image processing uses OpenCV and PIL
- Data augmentation uses Albumentations library
- HuggingFace token is loaded from .env file
