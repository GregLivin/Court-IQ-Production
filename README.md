# CourtIQ
AI-Powered NBA Player Performance Analytics Platform

---

## What Is CourtIQ?

CourtIQ is an AI-powered NBA player performance analytics platform that analyzes historical and real-time player data to generate predictive insights for player prop trends and performance forecasting.

In simple terms: CourtIQ uses data science and AI to predict how NBA players are likely to perform in upcoming games.

---

## Core Purpose

CourtIQ is designed to:

- Analyze NBA player statistics
- Detect trends and performance patterns
- Predict likely outcomes (PTS, REB, AST, PRA, etc.)
- Visualize player performance data
- Provide risk-based insights (safe vs. aggressive plays)

---

## How CourtIQ Works (Technical Breakdown)

### 1) Data Collection

CourtIQ pulls data from:

- NBA APIs (`nba_api`)
- Historical game logs
- Last 5 / Last 10 game windows
- Advanced tracking stats (when available)

Example features include:

- Points
- Rebounds
- Assists
- Minutes played
- Usage rate
- Matchup history
- Defensive ratings
- Pace
- Player tracking metrics

### 2) Data Processing

Using:

- Python
- Pandas
- NumPy

The system:

- Cleans data
- Removes inconsistencies
- Handles missing values
- Calculates rolling averages
- Generates trend indicators

### 3) Predictive Modeling

Using:

- Regression models
- Moving averages
- Weighted scoring models
- Potential ML upgrades (e.g., Random Forest, XGBoost)

The model predicts:

- Expected stat output
- Confidence score
- Over/Under probability trends

### 4) Visualization Layer

Using:

- Streamlit (UI migration target)
- FastAPI (backend logic)
- CSV exports (for instructor/professor validation)

Planned and current product features:

- Player dropdown selector
- Last 5 games trend graph
- Risk meter
- Confidence rating
- Data download button

---

## What Makes CourtIQ Different?

Many sports tools only show raw stats.

CourtIQ goes further by:

- Interpreting trends
- Calculating probabilities
- Classifying risk levels
- Providing a foundation for a future subscription-based analytics product

---

## Tech Stack

- Python
- Pandas
- NumPy
- `nba_api`
- Streamlit
- FastAPI
- CSV export workflows

---

## Vision

CourtIQ is built as a practical analytics engine today, with a clear path to become a production-grade predictive intelligence platform for NBA performance forecasting.
