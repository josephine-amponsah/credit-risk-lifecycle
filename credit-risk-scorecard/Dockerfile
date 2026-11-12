FROM python:3.11-slim
WORKDIR /srv
RUN pip install --no-cache-dir fastapi==0.115.6 "uvicorn[standard]==0.32.1" shap==0.46.0 \
        scikit-learn==1.8.0 pandas==2.2.3 numpy==2.1.3 joblib==1.4.2 \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY api ./api
COPY artefacts ./artefacts
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]