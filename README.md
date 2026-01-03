# EEE-UID_PROJECT
Multi Layer Perceptron's Neural Network with Optimisation Algorithm for Green House Gas prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-green)

##  Project Overview
This repository contains the implementation of a machine learning framework designed to forecast **Greenhouse Gas (CO₂) emissions**. The project compares traditional Deep Learning approaches against hybrid optimization techniques to achieve higher prediction accuracy.

The implementation is based on the research concepts found in the paper *"Multi-layer perceptron's neural network with optimization algorithm for greenhouse gas forecasting systems"*. It specifically analyzes time-series emission data to predict future trends.

## Key Features
* **Data Preprocessing:** Automated handling of missing values and MinMax normalization for stable neural network training.
* **Model Comparison:** Implements and compares three distinct approaches:
    1.  **LSTM (Long Short-Term Memory):** For capturing temporal dependencies in time-series data.
    2.  **MLP (Multi-Layer Perceptron):** A standard feedforward neural network.
    3.  **PSO-Optimized MLP:** Uses **Particle Swarm Optimization (PSO)** to fine-tune weights/features, simulating the optimization strategies (like MCOA) discussed in the associated research.
* **Performance Metrics:** Evaluates models using Mean Squared Error (MSE) and R-squared ($R^2$) scores.

# Comparative Analysis of Hybrid Neural Networks for GHG Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Optimization](https://img.shields.io/badge/Algorithm-PSO%20Hybrid-purple)

## 📖 1. Project Abstract
This project implements and analyzes a hybrid machine learning framework for forecasting Greenhouse Gas (CO₂) emissions. It specifically addresses the non-linearity and chaotic nature of environmental time-series data. 

Based on the research paper *"Multi-layer perceptron's neural network with optimization algorithm for greenhouse gas forecasting systems"*, this repository compares three modeling approaches:
1.  **Long Short-Term Memory (LSTM):** A Recurrent Neural Network (RNN) optimized for temporal sequences.
2.  **Multi-Layer Perceptron (MLP):** A baseline feed-forward deep learning model.
3.  **PSO-MLP (Hybrid):** An MLP integrated with **Particle Swarm Optimization (PSO)** to dynamically optimize input feature weights before training.

## 🏗️ 2. Technical Architecture & Methodology

The system follows a standard ML pipeline: **Data Ingestion $\rightarrow$ Normalization $\rightarrow$ Feature Optimization (Hybrid only) $\rightarrow$ Training $\rightarrow$ Evaluation**.

### A. The Data Pipeline
* **Input:** `EEEdataset_processed.csv` containing historical emission data (Years 1970–2023).
* **Preprocessing:** * Missing values are handled via zero-imputation (can be improved to mean/interpolation).
    * **Normalization:** `MinMaxScaler` is applied to scale values between $[0, 1]$. This is mathematically critical for Neural Networks to prevent exploding gradients and ensure the optimizers (Adam) converge efficiently.

### B. Model Theoretical Frameworks

#### 1. Long Short-Term Memory (LSTM)
LSTMs are designed to solve the *Vanishing Gradient Problem* inherent in standard RNNs. They utilize a gating mechanism:
* **Forget Gate:** Decides what information to discard from the cell state.
* **Input Gate:** Decides which new values to update.
* **Output Gate:** Decides what to output based on the cell state.
* *Application:* Used here to capture the year-over-year temporal dependencies of CO₂ emissions.



#### 2. Multi-Layer Perceptron (MLP)
A standard Deep Feed-Forward Network. 
* **Structure:** Input Layer $\rightarrow$ Hidden Layers (Dense + ReLU activation) $\rightarrow$ Output Layer (Linear activation for regression).
* **Limitation:** Standard MLPs treat all input features with equal initial randomness and rely solely on Backpropagation (Gradient Descent) to find relationships. They often get stuck in **Local Minima**.


#### 3. The Hybrid Approach: PSO-Optimized MLP
This is the core contribution of the project (aligned with the MCOA concept in the `EEE.pdf` paper).

**The Engineering Logic:**
Instead of feeding raw data into the MLP, we use **Particle Swarm Optimization (PSO)** to perform *Feature Weighting*.
1.  **Swarm Initialization:** A population of "particles" is created. Each particle represents a vector of weights (one weight per input feature).
2.  **Objective Function:** * The code applies the particle's weights to the training data ($X_{weighted} = X \cdot W$).
    * A temporary MLP is trained on this weighted data.
    * The validation loss (MSE) is returned as the "cost".
3.  **Update Rule:** Particles move toward the global best position (lowest MSE).
4.  **Result:** The PSO finds the optimal "importance" of every historical data point *before* the final model is fully trained. This acts as a powerful, non-linear feature selection mechanism.


## ⚙️ Installation & Execution

### Prerequisites
* Python 3.8+
* Libraries: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `pyswarm`, `matplotlib`

### Setup
1.  Clone the repo:
    ```bash
    git clone <repo_url>
    ```
2.  Install dependencies:
    ```bash
    pip install numpy pandas tensorflow scikit-learn pyswarm matplotlib
    ```

### Running the Experiment
1.  Open `eee_uid.py`.
2.  **CRITICAL:** Update line 15 to point to your local dataset location:
    ```python
    # file_path = r"C:\Users\aparn\Downloads\EEEdataset_processed.csv"  <-- OLD
    file_path = "EEEdataset_processed.csv"                              <-- NEW (Relative path)
    ```
3.  Run the script:
    ```bash
    python eee_uid.py
    ```

## Evaluation Metrics
The project evaluates performance using two standard regression metrics:

1.  **Mean Squared Error (MSE):** Measures the average squared difference between the estimated values and the actual value.
    $$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$
    *(Lower is better)*

2.  **R-squared ($R^2$):** Represents the proportion of variance for a dependent variable that's explained by an independent variable.
    $$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
    *(Closer to 1.0 is better)*

## 🔮 Future Improvements (Engineering Roadmap)
* **Algorithm Update:** The current code uses PSO (`pyswarm`). To strictly adhere to the `EEE.pdf` paper, implement the **Modified Coyote Optimization Algorithm (MCOA)** from scratch.
* **Hyperparameter Tuning:** Expand the PSO scope to optimize the *number of neurons* and *learning rate*, not just input weights.
* **Cross-Validation:** Implement K-Fold cross-validation to ensure the PSO hasn't overfit to the specific train/test split.




