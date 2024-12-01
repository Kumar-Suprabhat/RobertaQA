# Data Preparation and Fine-Tuning Pipeline

## Overview
This repository contains a robust and automated pipeline for data preparation and fine-tuning a Question Answering (QA) model. The pipeline integrates **GPT** for context and question generation, converts the data to **SQuAD format**, and fine-tunes a **RoBERTa model** for QA tasks. Additionally, the entire workflow is seamlessly tracked using **MLflow** for experiment management.

---

## Features
- **Data Conversion**: Converts input data from Excel files to JSON format.
- **Context and Question Generation**: Utilizes **GPT** to create meaningful context and multiple QA pairs for each data entry.
- **SQuAD Format Conversion**: Formats the generated data into the SQuAD dataset structure.
- **Model Fine-Tuning**: Fine-tunes a **RoBERTa-based Question Answering model** on the generated SQuAD dataset.
- **MLflow Integration**: Tracks experiments, hyperparameters, and metrics for efficient model management.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>

2. The data_preparation.py file is where the data is prepared
3. The Roberta_QA_finetuning.py file contains all the functions required in the fine tuning of the model
4. The main.py file is where we can run the entire script at one 

