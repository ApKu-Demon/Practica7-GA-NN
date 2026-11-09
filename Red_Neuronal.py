import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, roc_auc_score)

# CLASE CAPA ##################################################################
class Layer:
    def __init__(self, neurons: int, inputs: int, activation_function: callable, activation_derivative: callable, use_bias: bool):
        self.neurons = neurons
        self.inputs = inputs
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.use_bias = use_bias

        self.weights: np.ndarray = np.random.randn(inputs, neurons) * np.sqrt(2. / inputs)
        self.bias: np.ndarray = np.zeros(neurons) if use_bias else 0.0

        self.last_inputs = None
        self.last_y = None

    def feedForward(self, inputs: np.ndarray):
        self.last_inputs = inputs
        self.last_y = inputs @ self.weights + self.bias
        return self.activation_function(self.last_y)

    def backPropagation(self, grad_error: np.ndarray, learning_rate: float):
        delta = grad_error * self.activation_derivative(self.last_y)
        grad_W = self.last_inputs.T @ delta

        if self.use_bias:
            grad_b = np.sum(delta, axis=0)
            self.bias -= learning_rate * grad_b

        grad_error_prev = delta @ self.weights.T
        self.weights -= learning_rate * grad_W

        return grad_error_prev

# FUNCIONES DE ACTIVACION #####################################################
def relu(x: np.ndarray):
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray):
    return np.where(x > 0, 1.0, 0.0)

def linear(x: np.ndarray):
    return x

def linear_derivative(x: np.ndarray):
    return np.ones_like(x)

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x: np.ndarray):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x: np.ndarray):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x: np.ndarray):
    s = softmax(x)
    return s * (1 - s)

# RED NEURONAL (ARQUITECTURA) #################################################
class NeuralNetwork:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.feedForward(out)
        return out

    def backward(self, error, learning_rate):
        grad = error
        for layer in reversed(self.layers):
            grad = layer.backPropagation(grad, learning_rate)

    def train(self, X, y, epochs, learning_rate, target_loss=None, patience=None, min_delta=1e-5):
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            predictions = self.forward(X)
            
            # Derivada del MSE
            error = 2 * (predictions - y) / X.shape[0]
            self.backward(error, learning_rate)

            # Pérdida actual
            current_loss = np.mean((predictions - y) ** 2)

            # Impresión de progreso
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Época {epoch+1}/{epochs} - Pérdida (MSE): {current_loss:.6f}")

            # -------------------------------------------------------------
            # CONDICIÓN DE ALTO 1: Pérdida Objetivo Alcanzada
            # -------------------------------------------------------------
            if target_loss is not None and current_loss <= target_loss:
                print(f"\n¡Entrenamiento detenido -> Perdida optima alcanzada: MSE ({current_loss:.6f}) <= {target_loss} en época {epoch+1}.")
                break

            # -------------------------------------------------------------
            # CONDICIÓN DE ALTO 2: Parada Temprana por Paciencia (Paciencia)
            # -------------------------------------------------------------
            if patience is not None:
                if best_loss - current_loss > min_delta:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\n¡Entrenamiento detenido! La pérdida no mejoró durante {patience} épocas en la época {epoch+1}.")
                    break
                

# METRICAS REGRESION ##########################################################
def calcular_metricas_regresion(y_true, y_pred, nombres_salidas=None):
    """
    Calcula R, R2, SSE, MAE, MSE, RMSE por cada variable de salida
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        
    n_outputs = y_true.shape[1]
    metricas = []
    
    for i in range(n_outputs):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # Coeficiente de correlacion de Pearson (R)
        r_val = np.corrcoef(y_t, y_p)[0, 1] if len(np.unique(y_t)) > 1 else 0
        r2_val = r2_score(y_t, y_p)
        sse_val = np.sum((y_t - y_p) ** 2)
        mae_val = mean_absolute_error(y_t, y_p)
        mse_val = mean_squared_error(y_t, y_p)
        rmse_val = np.sqrt(mse_val)
        
        # Asignar nombre de la columna si se proporciona
        nombre_col = (
            nombres_salidas[i]
            if nombres_salidas and i < len(nombres_salidas)
            else f"Y{i+1}"
        )
        
        metricas.append({
            "Salida": nombre_col,
            "R": r_val,
            "R2": r2_val,
            "SSE": sse_val,
            "MAE": mae_val,
            "MSE": mse_val,
            "RMSE": rmse_val
        })
        
    return pd.DataFrame(metricas)

def calcular_metricas_multiclase(y_true_indices, y_pred_indices):
    acc = accuracy_score(y_true_indices, y_pred_indices)
    prec_macro = precision_score(y_true_indices, y_pred_indices, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)
    
    return pd.DataFrame([{
        "Métrica": "Exactitud (Accuracy)", "Valor": f"{acc * 100:.2f}%"
    }, {
        "Métrica": "Precisión (Promedio Macro)", "Valor": f"{prec_macro * 100:.2f}%"
    }, {
        "Métrica": "Precisión (Promedio Ponderado)", "Valor": f"{prec_weighted * 100:.2f}%"
    }])


def calcular_metricas_binaria(y_true, y_pred_prob):
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_class)
    
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except Exception:
        auc = 0.0
        
    prec = precision_score(y_true, y_pred_class, zero_division=0)
    
    return pd.DataFrame([{
        "Métrica": "Exactitud (Accuracy)", "Valor": f"{acc * 100:.2f}%"
    }, {
        "Métrica": "Área Bajo la Curva ROC (ROC-AUC)", "Valor": f"{auc:.4f}"
    }, {
        "Métrica": "Precisión", "Valor": f"{prec * 100:.2f}%"
    }])
        
        
        