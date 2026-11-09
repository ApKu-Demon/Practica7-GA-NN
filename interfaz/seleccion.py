import random

class SeleccionTorneo:
    def __init__(self, k=3):
        self.k = k
        
    def seleccion_poblacion(self, poblacion):
        mating_pool = []
        for _ in range(len(poblacion)):
            torneo = random.sample(poblacion, min(self.k, len(poblacion)))
            mejor = max(torneo, key=lambda c: c.score)
            mating_pool.append(mejor)
        return mating_pool
    
    
    