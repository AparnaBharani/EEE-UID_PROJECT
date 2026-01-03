# EEE-UID_PROJECT
Multi Layer Perceptron's Neural Network with Optimisation Algorithm for Green House Gas prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-green)

##  Project Overview
This repository contains the implementation of a machine learning framework designed to forecast **Greenhouse Gas (CO₂) emissions**. The project compares traditional Deep Learning approaches against hybrid optimization techniques to achieve higher prediction accuracy.

The implementation is based on the research concepts found in the paper *"Multi-layer perceptron's neural network with optimization algorithm for greenhouse gas forecasting systems"*. It specifically analyzes time-series emission data to predict future trends.

## 🚀Key Features
* **Data Preprocessing:** Automated handling of missing values and MinMax normalization for stable neural network training.
* **Model Comparison:** Implements and compares three distinct approaches:
    1.  **LSTM (Long Short-Term Memory):** For capturing temporal dependencies in time-series data.
    2.  **MLP (Multi-Layer Perceptron):** A standard feedforward neural network.
    3.  **PSO-Optimized MLP:** Uses **Particle Swarm Optimization (PSO)** to fine-tune weights/features, simulating the optimization strategies (like MCOA) discussed in the associated research.
* **Performance Metrics:** Evaluates models using Mean Squared Error (MSE) and R-squared ($R^2$) scores.


