import os
import numpy as np
import visualizaciones as vis

# 1. Importación limpia y agrupada del Algoritmo Genético gracias al __init__.py
from interfaz import AptitudNeuro, SeleccionTorneo, ReproduccionCruceSimple, MutacionHiperparametros

# 2. Importaciones de archivos en la raíz (hermanos del main.py)
from Red_Neuronal import Layer, NeuralNetwork, relu, relu_derivative, sigmoid, sigmoid_derivative, linear, linear_derivative, softmax, softmax_derivative
from GA import GA
from data_loader import cargar_dataset

# Catálogo de hiperparámetros
OPCIONES_LR = [0.001, 0.01, 0.05, 0.1]
OPCIONES_NEURONAS_L1 = [16, 32, 64, 128]
OPCIONES_NEURONAS_L2 = [8, 16, 32, 64]
LIMITES_CROMOSOMA = [(0, 3), (0, 3), (0, 3)] 

if __name__ == "__main__":
    print("=" * 50)
    print("SELECCIONA EL DATASET A PROCESAR")
    print("=" * 50)
    print("1. Regresión de Trayectoria 3D (Drone Trajectory)")
    print("2. Clasificación Multiclase (IMU Sensor Log)")
    print("3. Detección Binaria (Tampering)")
    
    opcion = int(input("\nIngresa el número (1/2/3): "))
    
    # Ajustar segun ruta (importante)
    ruta_base = r'C:\Users\PC\Documents\Universidad_UABC\01_Academico\Semestre_08\ComputoSuave'
    
    # Cargar datos mediante el módulo
    X_train, X_test, y_train, y_test, tipo_problema, num_salidas = cargar_dataset(opcion, ruta_base)
    
    print(f"\nDatos cargados. Modo: {tipo_problema.upper()}")
    print(f"Train X: {X_train.shape} | Test X: {X_test.shape}")
    
    # 1. EVALUAR RED BASE DINÁMICA
    print("\n--- ENTRENANDO RED BASE ---")
    if tipo_problema == 'regresion':
        capa_salida = Layer(num_salidas, 16, linear, linear_derivative, use_bias=True)
    elif tipo_problema == 'multiclase':
        capa_salida = Layer(num_salidas, 16, softmax, softmax_derivative, use_bias=True)
    else:
        capa_salida = Layer(1, 16, sigmoid, sigmoid_derivative, use_bias=True)
        
    nn_base = NeuralNetwork([
        Layer(32, X_train.shape[1], relu, relu_derivative, use_bias=True),
        Layer(16, 32, relu, relu_derivative, use_bias=True),
        capa_salida
    ])
    
    nn_base.train(X_train, y_train, epochs=10000, learning_rate=0.01)
    
    # 2. INICIAR ALGORITMO GENÉTICO
    print("\n--- INICIANDO OPTIMIZACIÓN GENÉTICA ---")
    gaparams = {
        "Aptitud": AptitudNeuro(OPCIONES_LR, OPCIONES_NEURONAS_L1, OPCIONES_NEURONAS_L2),
        "Seleccion": SeleccionTorneo(3),
        "Reproduccion": ReproduccionCruceSimple(),
        "Mutacion": MutacionHiperparametros(0.1),
        "por_remplazo": 0.5,
        "max_generaciones": 3
    }
    
    algoritmo_ga = GA(5, LIMITES_CROMOSOMA, gaparams)
    algoritmo_ga.buscar_solucion(1e-4, {
        "datos": (X_train, X_test, y_train, y_test), 
        "tipo_problema": tipo_problema, 
        "num_salidas": num_salidas
    })

    mejor_cromosoma = max(algoritmo_ga.poblacion, key=lambda c: c.score)
    
    mejor_lr = OPCIONES_LR[mejor_cromosoma.genes[0]]
    mejor_l1 = OPCIONES_NEURONAS_L1[mejor_cromosoma.genes[1]]
    mejor_l2 = OPCIONES_NEURONAS_L2[mejor_cromosoma.genes[2]]
    
    print("\n=== RESULTADOS FINALES ===")
    print(f"Mejor Modelo GA | Score (Acc/NegMSE): {mejor_cromosoma.score:.4f} | (LR: {OPCIONES_LR[mejor_cromosoma.genes[0]]}, L1: {OPCIONES_NEURONAS_L1[mejor_cromosoma.genes[1]]}, L2: {OPCIONES_NEURONAS_L2[mejor_cromosoma.genes[2]]})")
    
    print("\n--- GENERANDO GRÁFICAS DE RESULTADOS ---")

    print("\n--- ENTRENANDO LA MEJOR RED PARA EVALUACIÓN FINAL ---")
    if tipo_problema == 'regresion':
        capa_salida_final = Layer(num_salidas, mejor_l2, linear, linear_derivative, use_bias=True)
    elif tipo_problema == 'multiclase':
        capa_salida_final = Layer(num_salidas, mejor_l2, softmax, softmax_derivative, use_bias=True)
    else:
        capa_salida_final = Layer(1, mejor_l2, sigmoid, sigmoid_derivative, use_bias=True)

    mejor_modelo = NeuralNetwork([
        Layer(mejor_l1, X_train.shape[1], relu, relu_derivative, use_bias=True),
        Layer(mejor_l2, mejor_l1, relu, relu_derivative, use_bias=True),
        capa_salida_final
    ])
    
    mejor_modelo.train(X_train, y_train, epochs=10000, learning_rate=mejor_lr)

    # IMPORTAMOS LAS FUNCIONES DE MÉTRICAS QUE AGREGASTE EN Red_Neuronal.py
    from Red_Neuronal import calcular_metricas_regresion, calcular_metricas_multiclase, calcular_metricas_binaria

    y_pred = mejor_modelo.forward(X_test)

    print("\n=== MÉTRICAS FINALES DEL MEJOR MODELO ===")
    if tipo_problema == 'regresion':
        metricas_df = calcular_metricas_regresion(y_test, y_pred, nombres_salidas=['North', 'East', 'Down'])
        print(metricas_df.to_string(index=False))
    
    elif tipo_problema == 'multiclase':
        y_test_idx = np.argmax(y_test, axis=1)
        y_pred_idx = np.argmax(y_pred, axis=1)
        metricas_df = calcular_metricas_multiclase(y_test_idx, y_pred_idx)
        print(metricas_df.to_string(index=False))
    
    elif tipo_problema == 'binaria':
        metricas_df = calcular_metricas_binaria(y_test, y_pred)
        print(metricas_df.to_string(index=False))

    print("\n--- GENERANDO GRÁFICAS DE RESULTADOS ---")
    vis.graficar_matriz_confusion(y_test, y_pred, tipo_problema)
    vis.graficar_resultados_regresion(y_test, y_pred, tipo_problema)