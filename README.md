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

## Project Overview

CourtIQ Production is a data-driven NBA Player Performance Prediction Engine designed to analyze historical player statistics and generate structured performance insights.

The system evaluates player trends using statistical analysis techniques and recent performance windows to support predictive modeling.

This project demonstrates applied data analytics, statistical processing, and structured data export using Python.

---

## Problem Statement

Sports analytics requires accurate interpretation of performance trends. Traditional box score analysis does not fully capture consistency, short-term variability, or projection-based evaluation.

CourtIQ addresses this by:

- Analyzing rolling statistical windows (Last 5–15 games)
- Measuring performance consistency and volatility
- Generating structured outputs for further predictive modeling

---

## System Capabilities

The system performs the following:

1. Retrieves NBA player statistical data  
2. Cleans and structures raw datasets  
3. Calculates rolling averages and performance indicators  
4. Analyzes short-term trend patterns  
5. Exports processed results in CSV format  

Key Metrics Evaluated:
- Points (PTS)
- Rebounds (REB)
- Assists (AST)
- PRA (Points + Rebounds + Assists)

---

## Technologies Used

- Python
- Pandas
- NumPy
- nba_api
- GitHub for version control

---

## Repository Structure

Court-IQ-Production/
│
├── data/          Raw and processed datasets  
├── notebooks/     Analytical development files  
├── models/        Core analytical logic  
├── exports/       Generated CSV outputs  
└── README.md  

---

## Learning Outcomes

This project demonstrates:

- Data preprocessing and cleaning techniques
- Statistical trend analysis
- Structured CSV export validation
- Applied analytics in a real-world domain
- Collaborative development using GitHub

---

## Conclusion

CourtIQ Production provides a structured framework for NBA player performance analysis. The system establishes a foundation for future integration of machine learning models and predictive analytics enhancements.
