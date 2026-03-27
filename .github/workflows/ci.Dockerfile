FROM python:3.13-slim

RUN pip install --no-cache-dir \
    "setuptools>=61.0" \
    "pytest>=7.0.0" \
    "pytest-cov>=4.0.0" \
    "trustme>=1.0.0" \
    "cryptography>=43.0.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.25.0" \
    "ruff>=0.4.0"

WORKDIR /app
