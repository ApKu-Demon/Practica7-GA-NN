import random
import copy

class ReproduccionCruceSimple:
    def reproduccion_poblacion(self, nueva_gen, mating_pool, num_inv):
        while len(nueva_gen) < num_inv:
            p1, p2 = random.sample(mating_pool, 2)
            c1 = copy.deepcopy(p1)
            c2 = copy.deepcopy(p2)
            punto_cruce = random.randint(1, c1.n_genes - 1) if c1.n_genes > 1 else 1
            
            c1.genes = p1.genes[:punto_cruce] + p2.genes[punto_cruce:]
            c2.genes = p2.genes[:punto_cruce] + p1.genes[punto_cruce:]
            
            nueva_gen.append(c1)
            if len(nueva_gen) < num_inv:
                nueva_gen.append(c2)
                
                
                