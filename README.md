# CourtIQ Production

## NBA Player Performance Prediction Engine

**Course:** ITAI-2277  
**Instructor:** Prof. Sitaram Ayyagari  

**Team Members:**
- Gregory Livingston  
- Erwin Cheng  
- Heather Rathnam  

---

## 📌 Project Overview

CourtIQ Production is a data driven NBA Player Performance Prediction Engine that analyzes historical player statistics and generates structured performance projections.

The system evaluates recent performance trends using rolling statistical windows and exposes predictions through a live FastAPI backend.

This project demonstrates applied data analytics, backend API development, and cloud deployment in a production style environment.

---

## 🎯 Problem Statement

Traditional box score analysis does not fully capture short term trends, performance consistency, or projection based evaluation.

CourtIQ addresses this by:

- Analyzing rolling performance windows (Last 5 to 15 games)  
- Measuring short term performance patterns  
- Generating structured predictive outputs  
- Providing scalable backend API access  

---

## ⚙️ System Capabilities

The system performs the following:

1. Retrieves NBA player statistical data  
2. Cleans and structures raw datasets  
3. Calculates rolling averages and trend indicators  
4. Generates projected performance outputs  
5. Exposes predictions through a production API  

### Key Metrics Evaluated

- Points (PTS)  
- Rebounds (REB)  
- Assists (AST)  
- PRA (Points + Rebounds + Assists)  

---

## 🧠 Technology Stack

- Python  
- Pandas  
- NumPy  
- FastAPI  
- Uvicorn  
- GitHub  
- Render (Cloud Deployment)  

---

## 🗂 Repository Structure
Court-IQ-Production/
│
├── api/ FastAPI application and endpoints
├── src/ Core prediction logic
├── data/ Raw and processed datasets
├── notebooks/ Research and experimentation
├── tests/ Testing modules
├── requirements.txt
└── README.md


---

## 🌍 Live API (Production Deployment)

The FastAPI backend is deployed on Render:

https://court-iq-production-1.onrender.com/docs  

### How to Test the API

1. Open the link above  
2. Click `/predict`  
3. Select “Try it out”  
4. Enter a player name (example: Stephen Curry)  
5. Click Execute  

This is currently the backend service used for testing and validation.  
A user facing frontend dashboard will be built next.

---

## 💻 How to Run Locally

### 1️⃣ Clone the Repository 

```bash
git clone https://github.com/GregLivin/Court-IQ-Production.git
cd Court-IQ-Production

2️⃣ Create and Activate Virtual Environment
Windows: python -m venv .venv
.venv\Scripts\activate

Mac / Linux:
python -m venv .venv
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the API
uvicorn api.main:app --reload

Open in browser: http://127.0.0.1:8000/docs

🎓 Learning Outcomes

This project demonstrates:

Data preprocessing and structured dataset management

Rolling statistical analysis

Backend API design and deployment

Cloud hosting and production testing

Collaborative GitHub workflow

🚀 Future Roadmap

Build Streamlit frontend dashboard

Enhance prediction methodology

Add validation and evaluation metrics

Improve model sophistication beyond rolling averages



