# CourtIQ Production
NBA Player Performance Prediction Engine

Course: ITAI-2277  
Instructor: Prof. Sitaram Ayyagari  
Submission Date: January 28, 2025  

Team Members:
- Gregory Livingston
- Erwin Cheng
- Heather Rathnam

---
CourtIQ Production
PROJECT OVERVIEW

CourtIQ Production is a machine learning–powered NBA Player Performance Prediction Platform that analyzes historical player statistics and generates real-time projections.

The system combines statistical trend analysis with trained regression models and delivers predictions through an interactive Streamlit dashboard.

PROBLEM STATEMENT

Traditional box score analysis does not fully capture:

Short-term performance trends

Volatility and consistency

Projection-based forecasting

CourtIQ improves this by:

Analyzing rolling statistical windows (Last 5–15 games)

Engineering predictive features (minutes, usage, efficiency trends)

Training regression models for forecasting

Delivering structured predictions through a live dashboard

SYSTEM CAPABILITIES

Data Collection

Retrieves NBA player statistics using nba_api

Data Processing

Cleans and structures datasets

Engineers rolling averages and predictive features

Machine Learning

Trains regression models for performance prediction

Evaluates accuracy using MAE and RMSE

Deployment

Serves real-time predictions via Streamlit

Exports structured CSV outputs

KEY METRICS PREDICTED

Points (PTS)

Rebounds (REB)

Assists (AST)

PRA (Points + Rebounds + Assists)

TECHNOLOGIES USED

Python

Pandas

NumPy

scikit-learn

nba_api

Streamlit

GitHub

REPOSITORY STRUCTURE

Court-IQ-Production/
│
├── src/ Core application and ML logic
│ └── courtiq/ Models, loaders, prediction modules
├── data/ Raw and processed datasets
├── models/ Saved trained model artifacts
├── notebooks/ EDA and experimentation
├── exports/ Generated prediction outputs
├── requirements.txt Project dependencies
└── README.md

Learning Outcomes

This project demonstrates:

End-to-end data pipeline development

Feature engineering for sports analytics

Supervised machine learning implementation

Model evaluation and performance validation

Deployment of ML applications using Streamlit

Collaborative version control using GitHub

Conclusion

CourtIQ Production has evolved from a statistical analysis tool into a machine learning–driven predictive analytics platform. It establishes a scalable foundation for SaaS expansion, advanced modeling techniques, and future AI-powered sports intelligence applications.
