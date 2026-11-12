# Data

Source: [Lending Club accepted loans 2007–2018Q4](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (`accepted_2007_to_2018Q4.csv`, about 1.6 GB). It is not committed to the repo.

1. Download the CSV into this folder.
2. Split it into quarterly parquet files with `notebooks/app_data.ipynb` (output: `notebooks/data_splits/`, git-ignored).
3. Build the model and dashboard data:

```bash
python scripts/train_pd_model.py --src notebooks/data_splits --scored-dst build/scored
python scripts/prepare_data.py --src build/scored --dst dashboard/backend/app/data
```

The dashboard's own data (`dashboard/backend/app/data/`, about 75 MB) is committed so the hosted app can deploy straight from the repo.