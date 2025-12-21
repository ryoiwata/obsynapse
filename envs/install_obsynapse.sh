#!/bin/bash
# obsynapse Environment Setup Script
# This script creates a conda environment with all dependencies needed for the obsynapse project

# Deactivate any currently active conda environment
conda deactivate

# Create conda environment with Python 3.12 (compatible with DBT)
# Using local path ./obsynapse instead of global environment
conda create -p ./obsynapse python=3.12 --yes

# Activate the newly created environment
conda activate ./obsynapse