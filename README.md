# E-Commerce Retention Analysis

Can a visitor's first 14 days of activity help rank who returns during days 15-44? Using Retailrocket events, this project compares logistic regression and XGBoost with simple sorting rules and tests which feature groups help.

## Results

The cohort contains 991,833 visitors with complete follow-up; 3.5% returned during the outcome window. Validation selected XGBoost with all features and 200 trees by average precision (AP): 0.11734, compared with 0.11181 for logistic regression and 0.09841 for recency alone.

Removing recency and its related features caused the largest validation AP drop. Removing intent had little effect, and tuning improved XGBoost AP by only 0.00053.

On the later holdout of 114,973 visitors, the return rate was 2.7%:

| Ranking | AP | Capture at 5% | Capture at 10% | Capture at 20% |
| --- | --- | --- | --- | --- |
| XGBoost, 200 trees | 0.10299 | 23.8% | 31.0% | 40.9% |
| Recency | 0.08690 | 23.5% | 31.1% | 40.5% |
| Active days | 0.07250 | 21.8% | 28.7% | 36.6% |
| Event count | 0.05413 | 14.6% | 23.3% | 37.6% |
| High-intent count | 0.03005 | 7.3% | 12.2% | 21.9% |

Capture is the share of all returners found among the selected visitors. XGBoost improved overall ranking, but recency alone was competitive at these selection sizes and slightly ahead at 10%. These results do not show which visitors would benefit from outreach.

## Prediction setup

All windows start at each visitor's first observed event, which may not be their first-ever visit.

| Part | Definition |
| --- | --- |
| Features | Volume, activity patterns, intent, and recency during the first 14 elapsed days. |
| Scoring gap | The next 24 hours, excluded from features and outcome. |
| Outcome | Any recorded event from 15 elapsed days up to, but excluding, 45 elapsed days. |
| Eligibility | At least 45 days of follow-up before the log ends. |

The 14-day feature window and one-day gap are fixed experiment choices. A rolling 14-day window can touch 15 UTC calendar dates.

## Evaluation

Notebook 02 runs these comparisons on the same validation period:

1. Four simple rules: more events, more active days, more high-intent actions, or more recent activity at day 14.
2. Logistic regression and XGBoost with the full feature set. The combined intent count is omitted because it equals cart plus transaction counts; other correlated features are kept.
3. Remove one feature group at a time from the model with higher full-feature validation AP. Remove direct derivatives too: for example, events per active day is removed with volume or activity patterns.
4. For each algorithm, take its best tested feature set and try a few settings: two regularization values for logistic regression and three tree-setting changes for XGBoost. Defaults remain eligible.
5. Choose the rule or model, features, and settings with the highest validation AP. Refit that choice and evaluate it on the later period in a separate section.

The groups cover volume, activity patterns, intent, and recency. Notebook 02 defines their features. All features stop before day 14. Group removal measures added predictive value for this setup, not causal importance; related information can remain in other groups.

Training outcomes must be known by July 3, 2015. Validation scoring runs July 3-7; its outcomes are complete by August 8, when holdout scoring begins. The chosen model is refit using labels available by August 8. Waiting for outcomes leaves short evaluation periods within the 139-calendar-day log. All dates use UTC.

AP is the primary metric. Capture at 5%, 10%, and 20% measures the share of all returners found at each selection size. The return rate is a random-ranking AP reference. For tied scores, metrics average over random selection at the cutoff.

The holdout was inspected during earlier work, so it provides an additional evaluation rather than an untouched test. Features and settings are selected on validation. Measuring campaign impact would require a randomized experiment.

## Setup and execution

```bash
git clone https://github.com/chanethan408/ecommerce-retention-analysis.git
cd ecommerce-retention-analysis
conda env create -f environment.yml
conda activate ecommerce-retention-analysis
```

Download `events.csv` from the [Kaggle Retailrocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?resource=download&select=events.csv) into `data/raw/`.

From the project root:

```bash
python src/data_pipeline.py
python -m pytest
jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_eda.ipynb`: inspect activity, data quality, and observation coverage; define the prediction windows.
2. `notebooks/02_return_prediction.ipynb`: rebuild the cohort, compare rankings on validation data, and review the validation chart before choosing a ranking. Run the separate holdout section once the choice is settled.

`src/train_model.py` holds shared models, date groups, and metrics. The notebook runs validation comparisons; `src/evaluate_holdout.py` refits and evaluates the chosen ranking only when called from the holdout section. Running all notebook cells will also run that section.

The tests check event-window limits, enough follow-up time, UTC active dates, repeat runs, invalid input, tied scores, and which visitors belong in each training or evaluation period.
