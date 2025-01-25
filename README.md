# Explainable AI for Binary Classification

This project demonstrates how to train a binary classification model using PyTorch and explain its predictions using **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations). The goal is to understand the model's behavior and provide interpretability for individual predictions and global feature importance.

**Note:** The LIME implementation is currently being refined. Updates will follow in the coming days.

---

## Features

1. **Model Training:**
   - Fully connected neural network using PyTorch.
   - Supports multiple hidden layers and ReLU activations.
   - Saves the trained model and scaler for future use.

2. **Explainability:**
   - **SHAP**: Provides global and local explanations for predictions.
   - **LIME**: Explains individual predictions with feature importance.

3. **Visualizations:**
   - SHAP Summary Plots for global feature importance.
   - LIME Feature Importance Plots for specific predictions.

4. **Data Handling:**
   - Preprocessing with `StandardScaler` for feature normalization.
   - Easy integration with CSV datasets.

---

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/EXPLAINABLE_AI.git
   cd EXPLAINABLE_AI
   pip install -r requirements.txt



### Training the Model

2. To train the model, run:
    ```bash 
    python model/train.py
    

1. Load the training data (X_train.csv, y_train.csv) from the data/ directory.
2. Train a neural network using PyTorch.
3. Save the model checkpoint (model.pth) and a scaler for future inference.

### Generating Explanations

3. To generate explanations, run:
    ```bash
    python model/model_explainability.py
     

---


### Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. **Create a new branch** for your feature or bugfix.
3. **Submit a pull request** with a detailed description of your changes.



## Acknowledgments

- [SHAP](https://github.com/slundberg/shap) for model interpretability.
- [LIME](https://github.com/marcotcr/lime) for local explanations.
- [PyTorch](https://pytorch.org/) for the deep learning framework.


## Contact

For questions or feedback, please reach out to:

- **Razane Marref**  
- **Email**: [razane.marref07@gmail.com](razane.marref07@gmail.com)  
- **GitHub**: [RazaneMar](https://github.com/MarRazane)
