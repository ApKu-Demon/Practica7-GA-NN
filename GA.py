import copy
from typing import Dict
from Cromosoma import CromosomaHiperparametros

class GA(object):
    def __init__(self, num_inv: int, limites_genes: list, hyperparams: Dict):
        self.num_inv = num_inv
        self.poblacion = [CromosomaHiperparametros(limites_genes) for _ in range(num_inv)]
        self.hyperparams = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproduccion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams.get("por_remplazo", 0.5) * num_inv)
        self.max_generaciones = hyperparams.get("max_generaciones", 5)
        
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, tolerancia: float, var: Dict):
        avg_tol = 99999.0
        sum_score_prev = 0
        generacion = 1
        fail = False

        while not fail and generacion <= self.max_generaciones:
            print(f"\n--- INICIANDO GENERACIÓN {generacion} ---")
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            sum_scores = sum([c.score for c in self.poblacion]) / self.num_inv
            
            sum_score = sum_scores if sum_scores != 0 else 0.0001
            avg_tol = abs(sum_scores - sum_score_prev) / sum_score
            sum_score_prev = sum_scores

            print(f"Generacion {generacion:04d} | avg_tol: {avg_tol:.5f} | avg_score (Accuracy): {sum_scores:.5f}")

            if avg_tol < tolerancia:
                break

            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            
            if self.poblacion[0].score >= 0.999:
                print(f"\n¡ALTO! Solucion perfecta encontrada en la Generación {generacion}.")
                print("Deteniendo evolucion temprana...")
                break
            
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            
            self.reproduccion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)

            self.poblacion = copy.deepcopy(generacion_nueva)
            generacion += 1
            
            
            