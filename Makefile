.PHONY: monitoring-data monitoring-up scorecard-up test

# Rebuild the PD model and the dashboard data from the raw quarterly files
monitoring-data:
	cd credit-portfolio-risk-monitoring && \
	python scripts/train_pd_model.py --src notebooks/data_splits --scored-dst build/scored && \
	python scripts/prepare_data.py --src build/scored --dst dashboard/backend/app/data

monitoring-up:
	cd credit-portfolio-risk-monitoring/dashboard && cp -n .env.example .env; docker compose up --build

scorecard-up:
	cd credit-risk-scorecard && uvicorn api.main:app --reload --port 8001

test:
	cd credit-portfolio-risk-monitoring && pytest -q tests
	cd credit-risk-scorecard && pytest -q tests