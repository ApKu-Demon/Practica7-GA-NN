import random
from typing import List

class CromosomaHiperparametros(object):
    def __init__(self, limites: List[tuple]):
        self.n_genes = len(limites)
        self.limites = limites 
        self.score = -1.0
        self.genes = [0] * self.n_genes

    def llenar_random_data(self):
        for i in range(self.n_genes):
            self.genes[i] = random.randint(self.limites[i][0], self.limites[i][1])

    @property
    def representacion(self) -> str:
        return "-".join(str(g) for g in self.genes)
    
    
    