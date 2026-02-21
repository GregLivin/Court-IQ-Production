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
Project Overview

CourtIQ Production is a machine learning–powered NBA Player Performance Prediction Platform designed to analyze historical player statistics and generate real-time performance projections.

The system combines statistical trend analysis with trained regression models to produce structured, data-driven predictions through an interactive Streamlit dashboard.

This project demonstrates applied data engineering, machine learning modeling, and full-stack analytics deployment using Python.

Problem Statement

Traditional box score analysis does not account for short-term trends, volatility, or predictive forecasting.

CourtIQ addresses this by:

Analyzing rolling statistical windows (Last 5–15 games)

Engineering predictive features (minutes, recent averages, efficiency trends)

Training regression models to forecast player performance

Deploying predictions through an interactive analytics interface

System Capabilities

The system performs the following:

Retrieves NBA player statistical data using nba_api

Cleans and preprocesses datasets

Engineers rolling averages and trend-based features

Trains regression models for performance prediction

Evaluates model accuracy using MAE and RMSE

Serves real-time predictions through Streamlit

Exports structured results in CSV format

Key Metrics Predicted:

Points (PTS)

Rebounds (REB)

Assists (AST)

PRA (Points + Rebounds + Assists)

Technologies Used

Python

Pandas

NumPy

scikit-learn

nba_api

Streamlit (deployment & dashboard)

GitHub (version control)

Repository Structure

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
