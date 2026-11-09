import random

class MutacionHiperparametros:
    def __init__(self, prob=0.1):
        self.prob = prob
        
    def mutacion_poblacion(self, poblacion):
        for cromosoma in poblacion:
            for i in range(cromosoma.n_genes):
                if random.random() < self.prob:
                    cromosoma.genes[i] = random.randint(cromosoma.limites[i][0], cromosoma.limites[i][1])
                    
                    
                    