# E-Commerce Retention Analysis

## Overview
Analysis of e-commerce customer retention and churn.

## Setup

```bash
git clone https://github.com/chanethan408/ecommerce-retention-analysis.git
cd ecommerce-retention-analysis
conda env create -f environment.yml
conda activate ecommerce-retention-analysis
```

Download `events.csv` from the [Kaggle Retailrocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?resource=download&select=events.csv) and save it as `data/raw/events.csv`.

## Exploratory Data Analysis

Run `jupyter lab notebooks/01_eda.ipynb` from the project root and execute the notebook cells to reproduce the EDA.

The EDA finds 2.76 million events from 1.41 million visitors, with views accounting for 96.7% of events and 90% of visitors generating three events or fewer.

## Data Pipeline

From the project root, run:

```bash
python src/data_pipeline.py
```

The pipeline reads `data/raw/events.csv` and saves the retention cohort to `data/processed/retention_cohort.parquet`.

## Tests

Run the pipeline tests from the project root with:

```bash
python -m pytest
```

## Project Structure

```text
.
  .gitignore
  README.md
  environment.yml
src/
  data_pipeline.py
data/
  raw/
    events.csv  # download locally; not tracked by Git
  processed/
notebooks/
  01_eda.ipynb
tests/
  test_data_pipeline.py
```
