import numpy as np
from Red_Neuronal import Layer, NeuralNetwork, relu, relu_derivative, sigmoid, sigmoid_derivative, softmax, softmax_derivative, linear, linear_derivative, calcular_metricas_binaria

class AptitudNeuro:
    def __init__(self, opciones_lr, opciones_l1, opciones_l2):
        self.opciones_lr = opciones_lr
        self.opciones_l1 = opciones_l1
        self.opciones_l2 = opciones_l2

    def evaluar_poblacion(self, poblacion, var_entorno):
        X_train, X_test, y_train, y_test = var_entorno["datos"]
        tipo_problema = var_entorno["tipo_problema"]
        num_salidas = var_entorno["num_salidas"]
        input_size = X_train.shape[1]

        for i, cromosoma in enumerate(poblacion):
            lr = self.opciones_lr[cromosoma.genes[0]]
            n_l1 = self.opciones_l1[cromosoma.genes[1]]
            n_l2 = self.opciones_l2[cromosoma.genes[2]]
            
            if tipo_problema == 'regresion':
                capa_salida = Layer(num_salidas, n_l2, linear, linear_derivative, use_bias=True)
            elif tipo_problema == 'multiclase':
                capa_salida = Layer(num_salidas, n_l2, softmax, softmax_derivative, use_bias=True)
            else:
                capa_salida = Layer(1, n_l2, sigmoid, sigmoid_derivative, use_bias=True)

            nn = NeuralNetwork([
                Layer(n_l1, input_size, relu, relu_derivative, use_bias=True),
                Layer(n_l2, n_l1, relu, relu_derivative, use_bias=True),
                capa_salida
            ])

            nn.train(X_train, y_train, epochs=150, learning_rate=lr)
            preds = nn.forward(X_test)
            
            # Evaluación de aptitud según el problema
            if tipo_problema == 'regresion':
                mse = np.mean((preds - y_test)**2)
                score = -mse # Volvemos el MSE negativo para que el GA siga buscando "maximizar" el número
                print(f"  Individuo {i+1} | LR: {lr}, L1: {n_l1}, L2: {n_l2} -> MSE: {mse:.4f}")
            elif tipo_problema == 'multiclase':
                pred_indices = np.argmax(preds, axis=1)
                true_indices = np.argmax(y_test, axis=1)
                score = np.mean(pred_indices == true_indices)
                print(f"  Individuo {i+1} | LR: {lr}, L1: {n_l1}, L2: {n_l2} -> Accuracy: {score:.4f}")
            else:
                score = calcular_metricas_binaria(y_test, preds)
                print(f"  Individuo {i+1} | LR: {lr}, L1: {n_l1}, L2: {n_l2} -> Accuracy: {score:.4f}")

            cromosoma.score = score
            
            
            