import os
import torch
import shap
import pandas as pd
import numpy as np
#from lime import lime_tabular
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




class ExplainabilityNN:
    def __init__(self, model_state, input_size):
        # Match your model's exact layer names
        self.layer1_weight = model_state['layer1.0.weight']
        self.layer1_bias = model_state['layer1.0.bias']
        self.layer2_weight = model_state['layer2.0.weight']
        self.layer2_bias = model_state['layer2.0.bias']
        self.layer3_weight = model_state['layer3.0.weight']
        self.layer3_bias = model_state['layer3.0.bias']
        self.output_weight = model_state['output.weight']
        self.output_bias = model_state['output.bias']

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x @ self.layer1_weight.T + self.layer1_bias
        x = torch.relu(x)
        x = x @ self.layer2_weight.T + self.layer2_bias
        x = torch.relu(x)
        x = x @ self.layer3_weight.T + self.layer3_bias
        x = torch.relu(x)
        x = x @ self.output_weight.T + self.output_bias
        return torch.sigmoid(x).detach().numpy()     
    


def explain_model():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_path = os.path.join(project_root, 'data')
    model_path = os.path.join(project_root, 'model.pth')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize components
    model_state = checkpoint['model']
    scaler = checkpoint['scaler']
    
    # Load and preprocess data
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv')).values
    X_test_scaled = scaler.transform(X_test)
    
    # Create explainability model
    model = ExplainabilityNN(model_state, X_test.shape[1])

    # SHAP Explanation with stability fixes
    print("\nGenerating SHAP explanations...")
    
    # Use first sample as background
    background = X_test_scaled[:5]  
    explainer = shap.KernelExplainer(model.forward, background)
    
    # Calculate SHAP values for first sample
    shap_values = explainer.shap_values(X_test_scaled[0])
    
    # Handle binary classification output format
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  

    # Convert to proper array format
    shap_values = np.array(shap_values).reshape(1, -1)  
    
    # Create summary plot with explicit sorting
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]
    
    # Calculate mean absolute SHAP values for sorting
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    sort_inds = np.argsort(-mean_shap).astype(int) 
    
    # Create custom summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test_scaled[0:1],  
        feature_names=np.array(feature_names)[sort_inds],
        show=False
    )
    plt.title("SHAP Feature Importance for First Sample")
    plt.tight_layout()
    plt.show()

    # LIME Explanation
    #print("\nGenerating LIME explanation...")
    #lime_explainer = lime_tabular.LimeTabularExplainer(
    #    training_data=X_test_scaled,
    #    feature_names=feature_names,
    #    class_names=['No Diabetes', 'Diabetes'],
    #    mode='classification',
    #    discretize_continuous=True,
    #    random_state=42
    #)

# Explain first sample
    #explanation = lime_explainer.explain_instance(
    #    X_test_scaled[0],
    #    model.forward,
    #    num_features=5,
    #    top_labels=1  
    #)

# Plot explanation for class 1 (Diabetes)
    #fig = explanation.as_pyplot_figure(label=1)
    #fig.title("LIME Explanation for Diabetes Prediction")
    #fig.tight_layout()
    #plt.show()

if __name__ == "__main__":
    explain_model()