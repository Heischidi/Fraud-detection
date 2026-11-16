Dataset Instructions — European Credit-Card Fraud Detection
===========================================================

Why the dataset is not tracked in Git:
The raw file `creditcard.csv` is 150,828,752 bytes (~143.84 MiB / 150.83 MB), which exceeds
GitHub's 100 MiB single-file limit and is hosted externally by the original providers.

Where to obtain the dataset:
1. Visit the Kaggle ULB Credit Card Fraud Detection repository:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   (Source citation: Machine Learning Group - ULB, 2016; Dal Pozzolo et al., 2015).
2. Download and extract `creditcard.csv` into this folder (`data/creditcard.csv`).

Expected file properties:
- Expected size: 150,828,752 bytes (~143.84 MiB)
- Dimensions: 284,807 rows x 31 columns (`Time`, `V1`..`V28`, `Amount`, `Class`)
- No manual preprocessing is required before running `python analysis.py`.
