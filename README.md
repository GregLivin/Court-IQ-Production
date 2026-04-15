# CourtIQ Production

CourtIQ Production is a machine learning-powered NBA analytics engine that predicts player performance using rolling statistics and trend-based modeling.

Built with Python and FastAPI, this project transforms raw NBA data into real-time predictions through a scalable API.

## Live API

Deployed API docs:
https://courtiq-production.onrender.com/docs

## Features

- Player performance prediction for points, assists, and rebounds
- Rolling averages and trend analysis
- FastAPI backend for real-time predictions
- Cloud deployment for public API access
- Machine learning-ready architecture

## How It Works

1. Data collection
   - Historical NBA player stats

2. Feature engineering
   - Rolling averages from recent games
   - Trend indicators such as hot and cold streaks

3. Modeling
   - Regression-based prediction models

4. API layer
   - FastAPI serves predictions through endpoints

## Project Structure

```text
Court-IQ-Production/
│── api/            # FastAPI routes
│── src/            # Core logic for data processing and modeling
│── data/
│   └── raw/        # Raw datasets
│── app.py          # Main entry point
│── requirements.txt