# YuvaIntern Week 1 - Data Acquisition, Cleaning and Preprocessing

## Project
**Wine Dataset Cleaning and Preprocessing with Python**

This project completes the Week 1 YuvaIntern task by acquiring a public dataset,
checking its quality, detecting outliers, validating values, and producing a
cleaned dataset suitable for later analysis.

## Dataset
The project uses the UCI Machine Learning Repository Wine dataset. It contains
178 observations, 13 numeric input features, and a target class representing
three wine cultivars. The dataset is publicly available and licensed under
CC BY 4.0.

Source: https://archive.ics.uci.edu/dataset/109/wine

## Files
- `data/wine_raw.csv` - raw dataset used in the project
- `data/wine_cleaned.csv` - cleaned/preprocessed dataset
- `src/data_cleaning.py` - reproducible cleaning script
- `week1_wine_preprocessing.ipynb` - notebook version of the analysis
- `figures/` - generated charts used in the report
- `YuvaIntern_Week1_Report.docx` - final report

## Main preprocessing decisions
1. Checked data types and dataset dimensions.
2. Checked missing values.
3. Checked duplicate records.
4. Validated numeric feature values.
5. Detected outliers using the 1.5 × IQR rule.
6. Capped feature outliers at IQR fences rather than deleting rows, because
   extreme chemical measurements may be legitimate observations.
7. Saved the cleaned dataset for downstream analysis.

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
python src/data_cleaning.py
jupyter notebook week1_wine_preprocessing.ipynb
```

## GitHub
Create a public GitHub repository and upload the contents of this folder.
Then paste your repository URL into the YuvaIntern submission form.
