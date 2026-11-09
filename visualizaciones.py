import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# ==========================================
# GRÁFICAS PARA CLASIFICACIÓN
# ==========================================
def graficar_matriz_confusion(y_true, y_pred, tipo_problema):
    if tipo_problema == 'regresion':
        return

    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = np.round(y_pred).astype(int)
        y_true = y_true.astype(int)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('1. Matriz de Confusión (Heatmap)')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.show()


# ==========================================
# GRÁFICAS PARA REGRESIÓN (Mínimo 4)
# ==========================================
def graficar_resultados_regresion(y_true, y_pred, tipo_problema):
    """
    Despliega 4 gráficas distintas para analizar problemas de regresión.
    """
    if tipo_problema != 'regresion':
        return

    # GRÁFICA 1: Dispersión (La que ya tenías)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, color='orange', label='Predicción vs Real')
    min_val = min(y_true[:, 0].min(), y_pred[:, 0].min())
    max_val = max(y_true[:, 0].max(), y_pred[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Alineación Perfecta')
    plt.title('1. Dispersión: Valor Real vs Predicción (Dim 1)')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # GRÁFICA 2: Series de Tiempo (Aquí verás DOS COLORES DISTINTOS)
    plt.figure(figsize=(10, 5))
    muestras = 100 # Graficamos solo los primeros 100 datos para que sea legible
    plt.plot(y_true[:muestras, 0], label='Valor Real', color='blue', marker='o', markersize=4)
    plt.plot(y_pred[:muestras, 0], label='Predicción', color='orange', marker='x', markersize=4)
    plt.title('2. Comparativa Directa: Real vs Predicción (Primeros 100 datos)')
    plt.xlabel('Muestra de Tiempo')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # GRÁFICA 3: Histograma de Errores (Residuales)
    plt.figure(figsize=(8, 6))
    errores = y_true[:, 0] - y_pred[:, 0]
    sns.histplot(errores, bins=50, kde=True, color='purple')
    plt.title('3. Distribución del Error (Residuales)')
    plt.xlabel('Error (Real - Predicción)')
    plt.ylabel('Frecuencia')
    plt.axvline(x=0, color='red', linestyle='--', label='Error Cero')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # GRÁFICA 4: Comparación entre dimensiones (Multi-salida)
    # Solo si el modelo predice más de una dimensión (ej. North, East, Down)
    if y_true.shape[1] >= 3:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        nombres = ['North (Dim 1)', 'East (Dim 2)', 'Down (Dim 3)']
        for i in range(3):
            axs[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.3, color='teal')
            min_v = min(y_true[:, i].min(), y_pred[:, i].min())
            max_v = max(y_true[:, i].max(), y_pred[:, i].max())
            axs[i].plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--')
            axs[i].set_title(nombres[i])
            axs[i].set_xlabel('Real')
            axs[i].set_ylabel('Pred')
            axs[i].grid(True, alpha=0.3)
        plt.suptitle('4. Rendimiento por cada Eje 3D (Subplots)')
        plt.tight_layout()
        plt.show()


# ==========================================
# GRÁFICA DEL ALGORITMO GENÉTICO
# ==========================================
def graficar_evolucion_ga(historial_generaciones, historial_scores):
    if not historial_generaciones or not historial_scores:
        return
    plt.figure(figsize=(8, 6))
    plt.plot(historial_generaciones, historial_scores, marker='o', color='green', linestyle='-', linewidth=2)
    plt.title('Evolución del Algoritmo Genético')
    plt.xlabel('Generación')
    plt.ylabel('Mejor Score (Accuracy / -MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()