# AI Traffic Flow Prediction & Signal Optimisation
## Overview

This project develops a machine learning-based system for short-term traffic flow prediction and adaptive traffic signal optimisation. Using historical traffic data from multiple sensor locations, the model predicts future traffic conditions and translates them into intelligent signal timing decisions.

The system demonstrates a practical step toward AI-driven smart transportation systems and real-time urban mobility optimisation.

## Objectives
Predict short-term traffic flow using historical data
Model traffic behaviour across multiple sensor locations
Evaluate model performance using standard regression metrics
Convert predictions into actionable traffic signal recommendations
Demonstrate a foundation for intelligent transport systems

## Dataset

The project uses the Traffic Flow Forecasting Dataset from the UCI Machine Learning Repository Zhao, L. (2019). Traffic Flow Forecasting [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57897..

Traffic measured every 15 minutes
Data collected across 36 sensor locations
Includes historical traffic windows for prediction

## Methodology
1. Data Processing
Loaded MATLAB .mat dataset using scipy
Converted sparse matrices into dense arrays
Reshaped time-series data into machine learning format
Prepared multi-output regression targets
2. Feature Structure
Input: past traffic data across 36 sensors over 48 time steps
Output: predicted traffic across 36 sensors at next time step
3. Model

A Random Forest Regressor was used for multi-output prediction:

Handles non-linear relationships
Robust to noise
Suitable for high-dimensional data

## Results
Metric	Value
MAE	0.0316
RMSE	0.0471
R²	0.9141

The model explains over 91% of variance, indicating strong predictive performance.

## Visualisations
Traffic Prediction vs Actual

Error Distribution

## Signal Optimisation Layer

A simple decision system was implemented to convert predicted traffic into signal timing:

Traffic Level	Green Time
Low	30 sec
Moderate	45 sec
High	60 sec
Severe	90 sec

Example output is saved as:

outputs/sensor_0_signal_recommendations.csv

## System Insight

This project moves beyond prediction by introducing a decision-making layer, transforming the model into a basic intelligent traffic control system.

## Future Work
Integration with drone-based aerial sensing
Incorporation of connected vehicle (CAV) data
Multi-agent reinforcement learning for dynamic optimisation
Real-time deployment for smart city environments

## Relevance

This project aligns with research areas in:

Intelligent Transportation Systems (ITS)
Smart Cities
AI-driven infrastructure optimisation
Autonomous and connected vehicle ecosystems

## Tech Stack
Python
NumPy
Pandas
Scikit-learn
Matplotlib
SciPy

## Author
Nnamdi Onuigbo
AI Systems Engineer | Automation Architect
