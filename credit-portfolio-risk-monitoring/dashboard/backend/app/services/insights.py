"""Simple insights helpers for the Overview API endpoints.

Functions:
- load_data(period=None, data_url=...): load dataframe from the repo's data_splits (remote via GitHub API or raw) and return a pd.DataFrame
- summary(df): compute total lent, repaid, default rate, interest earned
- aggregates(df): group by loan purpose and compute lent/repaid/interest
- loan_purposes(df): return sorted unique loan purpose list

This module is intentionally lightweight and defensive: it tolerates missing columns
and falls back to 0/empty lists when an operation is not possible.
"""
from __future__ import annotations

import os
import io
import json
import requests
import pandas as pd
import pyarrow.parquet as pq
import functools
from pathlib import Path
from typing import Optional, List, Dict

# Prefer the local backend data directory. In this repo the parquet files live under
# dashboard/backend/app/data. Keep a repo-level fallback for local testing and deployment.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
FALLBACK_DATA_DIR = Path(__file__).resolve().parents[3] / 'data'
if not DEFAULT_DATA_DIR.exists() and FALLBACK_DATA_DIR.exists():
	DEFAULT_DATA_DIR = FALLBACK_DATA_DIR
DEFAULT_DATA_URL = str(DEFAULT_DATA_DIR)


def _read_bytes_to_df(content: bytes, name: str) -> pd.DataFrame:
	try:
		if name.endswith('.parquet'):
			return pd.read_parquet(io.BytesIO(content))
		# CSV or gzipped CSV
		return pd.read_csv(io.BytesIO(content), parse_dates=['issue_d'], low_memory=False)
	except Exception:
		# Last resort: try CSV without parse_dates
		try:
			return pd.read_csv(io.BytesIO(content), low_memory=False)
		except Exception:
			raise


def _normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
	for c in df.select_dtypes(include=['object']).columns:
		df[c] = df[c].apply(lambda v: None if pd.isna(v) else (v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)))
	return df


def _list_remote_datafiles(data_url: str, timeout: int = 10) -> List[str]:
	if not (data_url.startswith('https://github.com/') and '/tree/' in data_url):
		return []
	owner_repo, rest = data_url.replace('https://github.com/', '').split('/tree/', 1)
	branch, _, path = rest.partition('/')
	api_url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
	resp = requests.get(api_url, params={'ref': branch}, timeout=timeout)
	if resp.status_code != 200:
		return []
	items = resp.json()
	return [it['name'] for it in items if it.get('name', '').startswith('data_')]


# Only the columns the API uses. Reading all 150+ Lending Club columns costs ~350 MB per quarter.
USECOLS = [
	'id', 'issue_d', 'loan_status', 'last_pymnt_d', 'term', 'grade', 'sub_grade', 'purpose',
	'int_rate', 'funded_amnt', 'out_prncp', 'total_rec_prncp', 'total_rec_int', 'recoveries',
	'collection_recovery_fee', 'annual_inc', 'dti', 'fico_range_low', 'prob_default', 'pd_calibrated',
]
CATEGORICAL = ['loan_status', 'term', 'grade', 'sub_grade', 'purpose']


def list_periods(data_dir: Path = DEFAULT_DATA_DIR) -> List[str]:
	return sorted(n[len('data_'):].split('.')[0] for n in os.listdir(data_dir) if n.startswith('data_'))


def resolve_period(period: Optional[str]) -> Optional[str]:
	periods = list_periods()
	if not periods:
		return None
	return period if period in periods else periods[-1]


def load_data(period: Optional[str] = None) -> pd.DataFrame:
	"""Return the (read-only, cached) loan frame for a quarter. Callers must not mutate it."""
	resolved = resolve_period(period)
	return _load_period(resolved) if resolved else pd.DataFrame()


@functools.lru_cache(maxsize=1)   # keyed on the resolved period, so None and '2018Q4' share one entry
def _load_period(period: str) -> pd.DataFrame:
	path = DEFAULT_DATA_DIR / f'data_{period}.parquet'
	available = set(pq.read_schema(path).names)
	df = pd.read_parquet(path, columns=[c for c in USECOLS if c in available])
	for c in CATEGORICAL:
		if c in df.columns:
			df[c] = df[c].astype('category')
	df['loan_purpose'] = df['purpose'] if 'purpose' in df.columns else pd.Series(dtype='category')
	return df


def _ensure_loan_purpose_column(df: pd.DataFrame) -> pd.DataFrame:
	if 'loan_purpose' in df.columns:
		return df
	# common source column is 'purpose'
	if 'purpose' in df.columns:
		df = df.copy()
		df['loan_purpose'] = df['purpose']
		return df
	return df


def summary(df: pd.DataFrame) -> Dict[str, float]:
	"""Compute top-level summary metrics from a DataFrame.

	Returns dict with keys: total_lent, repaid, default_rate, interest_earned
	"""
	if df is None or df.empty:
		return {'total_lent': 0.0, 'repaid': 0.0, 'default_rate': 0.0, 'interest_earned': 0.0}
	df = _ensure_loan_purpose_column(df)
	total_lent = float(df.get('funded_amnt', pd.Series(dtype='float')).sum()) if 'funded_amnt' in df.columns else 0.0
	# repaid principal
	repaid = float(df.get('total_rec_prncp', pd.Series(dtype='float')).sum()) if 'total_rec_prncp' in df.columns else 0.0
	# interest earned
	interest = float(df.get('total_rec_int', pd.Series(dtype='float')).sum()) if 'total_rec_int' in df.columns else 0.0
	# Default rate: use notebook mapping to targets and exclude 'current' loans (label 2)
	default_rate = 0.0
	if 'loan_status' in df.columns:
		s = df['loan_status'].astype(str).fillna('')

		def _map_target(x: str) -> int:
			# follow notebook mapping: Fully Paid -> 0, Charged Off/Default/Late(31-120) ->1, else ->2
			if x == 'Fully Paid':
				return 0
			if x in ('Charged Off', 'Default', 'Late (31-120 days)'):
				return 1
			return 2

		try:
			targets = s.map(_map_target)
			non_current = targets != 2
			denom = int(non_current.sum())
			if denom > 0:
				default_rate = float(((targets == 1) & non_current).sum()) / denom
			else:
				default_rate = 0.0
		except Exception:
			default_rate = 0.0
	return {'total_lent': total_lent, 'repaid': repaid, 'default_rate': default_rate, 'interest_earned': interest}


def aggregates(df: pd.DataFrame) -> List[Dict[str, float]]:
	"""Return aggregates grouped by loan purpose with fields lent/repaid/interest.

	Output: list of dicts: { 'loan_purpose': str, 'lent': float, 'repaid': float, 'interest': float }
	"""
	if df is None or df.empty:
		return []
	df = _ensure_loan_purpose_column(df)
	group_col = 'loan_purpose' if 'loan_purpose' in df.columns else None
	if not group_col:
		return []
	# numeric columns with fallbacks
	lent_col = 'funded_amnt' if 'funded_amnt' in df.columns else None
	repaid_col = 'total_rec_prncp' if 'total_rec_prncp' in df.columns else None
	interest_col = 'total_rec_int' if 'total_rec_int' in df.columns else None

	agg = df.groupby(group_col).agg({
		lent_col: 'sum' if lent_col else pd.NamedAgg(column=group_col, aggfunc=lambda x: 0),
		repaid_col: 'sum' if repaid_col else pd.NamedAgg(column=group_col, aggfunc=lambda x: 0),
		interest_col: 'sum' if interest_col else pd.NamedAgg(column=group_col, aggfunc=lambda x: 0),
	})
	# normalize column names
	agg = agg.rename(columns={
		lent_col: 'lent' if lent_col else 'lent',
		repaid_col: 'repaid' if repaid_col else 'repaid',
		interest_col: 'interest' if interest_col else 'interest',
	})
	agg = agg.reset_index()
	# ensure numeric values
	rows = []
	for _, r in agg.iterrows():
		rows.append({'loan_purpose': r[group_col], 'lent': float(r.get('lent', 0.0) or 0.0), 'repaid': float(r.get('repaid', 0.0) or 0.0), 'interest': float(r.get('interest', 0.0) or 0.0)})
	# sort by lent desc
	rows = sorted(rows, key=lambda x: x['lent'], reverse=True)
	return rows


def loan_purposes(df: pd.DataFrame) -> List[str]:
	if df is None or df.empty:
		return []
	df = _ensure_loan_purpose_column(df)
	if 'loan_purpose' not in df.columns:
		return []
	opts = sorted(pd.Series(df['loan_purpose'].dropna().unique()).astype(str).tolist())
	return opts


