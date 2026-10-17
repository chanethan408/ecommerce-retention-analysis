# E-Commerce Retention Analysis

## Overview
Analysis of e-commerce customer retention and churn.

## Setup

```bash
conda env create -f environment.yml
conda activate retention-analytics
```

Download `events.csv` from the [Kaggle Retailrocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?resource=download&select=events.csv) and save it as `data/raw/events.csv`.

## Exploratory Data Analysis

Run `jupyter lab notebooks/01_eda.ipynb` from the project root and execute the notebook cells to reproduce the EDA.

The EDA finds 2.76 million events from 1.41 million visitors, with views accounting for 96.7% of events and 90% of visitors generating three events or fewer.

## Project Structure

```text
.
  .gitignore
  README.md
  environment.yml
data/
  raw/
    events.csv  # download locally; not tracked by Git
  processed/
notebooks/
  01_eda.ipynb
```
