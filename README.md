# E-Commerce Retention Analysis

Can a visitor's first 14 days of activity help rank who returns during days 15-44? This project tests that question with Retailrocket event data. It compares two activity rules with logistic regression and XGBoost using chronological validation and a later holdout.

Scores are made at day 14 and predict observed returns. Measuring the effect of outreach would require a randomized experiment.

## Results

The cohort contains 991,833 visitors with complete follow-up; 3.5% returned during the outcome window. Validation selected logistic regression by average precision (AP). On the later holdout of 114,973 visitors:

| Ranking | Average precision | Top-10% return rate | Share of returners captured |
| --- | --- | --- | --- |
| Logistic regression | 0.079 | 7.8% | 28.8% |
| Active days | 0.072 | 7.8% | 28.7% |
| Event count | 0.054 | 6.3% | 23.3% |

The holdout return rate was 2.7%. Logistic regression's top 10% had about 2.9 times that rate. It had the highest validation AP and exceeded both rules on holdout AP. At the illustrative 10% cutoff, active-day ranking performed similarly.

## Prediction setup

All windows start at each visitor's first observed event, which may not be their first-ever visit.

| Part | Definition |
| --- | --- |
| Features | Event count, distinct UTC active dates, and cart additions plus transactions during the first 14 elapsed days. |
| Scoring gap | The next 24 hours, excluded from features and outcome. |
| Outcome | Any recorded event from 15 elapsed days up to, but excluding, 45 elapsed days. |
| Eligibility | At least 45 days of follow-up before the log ends. |

The 14-day feature window and one-day gap are fixed experiment choices. A rolling 14-day window can touch 15 UTC calendar dates.

## Evaluation

Event-count and active-day rankings test whether a simple rule is enough. Logistic regression and XGBoost use all three features. The ranking with the highest validation average precision (AP) is selected before holdout evaluation.

Training outcomes must be complete by July 3, 2015. Validation scoring runs July 3-7; those outcomes are complete by August 8, when holdout scoring begins. A selected model is refit using labels available by August 8. Requiring complete outcomes before each stage leaves short evaluation periods within this 139-calendar-day log. All cutoffs use UTC.

AP measures ranking across thresholds; the return rate is a random-ranking reference. Top-10% precision and recall show how returns concentrate within an illustrative capacity limit. For tied scores, metrics average over random selection at the cutoff.

Activity-rule benchmarks were added after earlier holdout results had been inspected. The evaluation preserves chronological separation, but the final comparison was developed iteratively.

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
python src/train_model.py
python -m pytest
jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_eda.ipynb`: inspect activity, data quality, and observation coverage; define the prediction windows.
2. `notebooks/02_return_prediction.ipynb`: rebuild the cohort, check features and labels, compare rankings, and interpret holdout results.

The tests check event-window limits, enough follow-up time, UTC active dates, repeat runs, invalid input, tied scores, and which visitors belong in each training or evaluation period.
