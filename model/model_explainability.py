import torch
import shap
import numpy as np
import pandas as pd
from lime import lime_tabular
import matplotlib.pyplot as plt

class ExplainabilityNN:
    def __init__(self, checkpoint, input_size):
        self.layer1_weight = checkpoint['model.layer1.0.weight']
        self.layer1_bias = checkpoint['model.layer1.0.bias']
        self.layer2_weight = checkpoint['model.layer2.0.weight']
        self.layer2_bias = checkpoint['model.layer2.0.bias']
        self.layer3_weight = checkpoint['model.layer3.0.weight']
        self.layer3_bias = checkpoint['model.layer3.0.bias']
        self.output_weight = checkpoint['model.output.weight']
        self.output_bias = checkpoint['model.output.bias']
        self.input_size = input_size

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
        return torch.sigmoid(x)

def explain_model(data_path='../data', model_path='model.pth'):
    X_test = pd.read_csv(f'{data_path}/X_test.csv').values
    checkpoint = torch.load(model_path)
    scaler = checkpoint['scaler']
    
    input_size = X_test.shape[1]
    model = ExplainabilityNN(checkpoint, input_size)
    
    X_test_scaled = scaler.transform(X_test)
    
    # SHAP Explanation
    print("Generating SHAP explanations...")
    background = shap.kmeans(X_test_scaled, 10)
    explainer = shap.KernelExplainer(
        lambda x: model.forward(x).detach().numpy(),
        background
    )
    
    shap_values = explainer.shap_values(X_test_scaled[:100], nsamples=100)
    shap.summary_plot(shap_values, X_test_scaled, feature_names=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    print("__________________________________________________________")

    # LIME Explanation
    print("\nGenerating LIME explanations...")
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_test_scaled,
        feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        class_names=['No Diabetes', 'Diabetes'],
        mode='classification',
        discretize_continuous=False
    )
    
    exp = lime_explainer.explain_instance(
        X_test_scaled[0], 
        model.forward, 
        num_features=8,
        top_labels=1
    )
    
    # Plot LIME explanation
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    explain_model()