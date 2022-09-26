from qiskit import(QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, IBMQ)
import copy
import random
import math
import statistics as stat
import utils

ELITISM_R= "Reinforcement_elitism"
ELITISM_Q= "Quantistic_elitism"
ELITISM_D ="Deterministic_elitism"

def isBetter(fit_a, fit_b, optMax):
    if optMax==True:
        if fit_a > fit_b:
            return True
        return False
    else:
        if fit_a < fit_b:
            return True
        return False


class globalBest:
    def __init__(self):
        self.chr = []
        self.theta = []
        self.fitness = None
        self.gen = None

    def display(self):
        print("The best chromosome is: ", self.chr)
        print("Its fitness value is: ", self.fitness)
        print("The fitness evaluations to obtain the best: ", self.gen)

def fromQtoC(max_counts):
    max_counts = ''.join(reversed(max_counts))
    classical_chromosomes=max_counts.split(" ")
    return classical_chromosomes


def computeFitnesses(classical_chromosomes, fitness_f):
    fitnesses=[]
    for chr in classical_chromosomes:
        fitnesses.append(fitness_f(chr))
    return fitnesses

def computeBest(fitnesses, optMax):
    if optMax==True:
        best_fitness=max(fitnesses)
    else:
        best_fitness = min(fitnesses)
    return best_fitness, fitnesses.index(best_fitness)

def averageFitnessonPopulation(counts, num_populations, num_shots, fitness_f, num_gen=0,gauss=False):
    chr_keys=counts.keys()
    dict_counts_inv={}
    for i in chr_keys:
        k_inv=''.join(reversed(i))
        dict_counts_inv[k_inv]=counts[i]
    chr_keys=dict_counts_inv.keys()
    l=[]
    for i in range(num_populations):
        l.append({})
    for k in chr_keys:
        k_c = k.split(" ")
        for j in range(num_populations):
            if k_c[j] not in l[j]:
                l[j][k_c[j]] = dict_counts_inv[k]
            else:
                l[j][k_c[j]] = l[j].get(k_c[j])+(dict_counts_inv[k])
    average=[]
    for i in range(num_populations):
        #print(len(l[i]), ": length of the population of index", i)
        #print(max(l[i].values()), ": max probability of the population of index", i)
        #print(l[i])
        fitnesses=computeFitnesses(l[i].keys(), fitness_f)
        n=0
        avg=0
        #print("i",i)
        #print("sum values ", sum(l[i].values()))
        new_fit_list = []
        for m in l[i].keys():
            #print(m, "=", l[i][m],"*",fitnesses[n])
            avg+=l[i][m]/num_shots*fitnesses[n]
            new_fit_list += [fitnesses[n]] * l[i][m]
            n+=1
        average.append(avg)
        if gauss:
            mean_pop=stat.mean(new_fit_list)
            std_pop=stat.stdev(new_fit_list)
            x_min=min(new_fit_list)
            x_max=max(new_fit_list)
            #print(new_fit_list)
            utils.plotGaussian(str(num_gen)+"_"+str(i+1),sorted(new_fit_list), x_min,x_max,mean_pop,std_pop)

    return average

def setupCircuit(number_of_populations, gene_per_chromosome):
    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit()

    for i in range(number_of_populations):
        circuit.add_register(QuantumRegister(gene_per_chromosome))

    for i in range(number_of_populations):
        circuit.add_register(ClassicalRegister(gene_per_chromosome))

    return circuit

def resetCircuit(circuit):
    list_obj=copy.deepcopy(circuit.data)
    for obj in list_obj:
        circuit.data.remove(obj)

def getMaxProbKey(counts):
    return max(counts, key = lambda k: counts[k])

def create_dict_chr(circuit, classical_chromosomes):
    dict_chr={}
    chr="".join(classical_chromosomes)
    #print(chr)
    i=0
    for quantum_register in circuit.qregs:
        for qubit in quantum_register:
            dict_chr[qubit]=chr[i]
            i+=1
    return dict_chr

def initializeTheta(circuit,epsilon_init):
    dict={}
    for quantum_register in circuit.qregs:
        for qubit in quantum_register:
            if random.random() < 0.5:
                theta = epsilon_init
            else:
                theta = - epsilon_init
            dict[qubit] = theta
    return dict

def applyMutationOnListWithinRange(circuit, prob, list_mutation, theta):
    for qubit in list_mutation:
        r= random.random()
        if r < prob:
            #print("I am here", r, prob)
            rot_amount=2*theta[qubit]
            circuit.ry(-rot_amount, qubit)
            theta[qubit]=theta[qubit]-rot_amount


def applyMultiRotationOnList(circuit, theta, list_qubit):
    for qubit in list_qubit:
        circuit.ry(theta[qubit], qubit)


def applyEntanglementOnList(circuit, index_best, list_entang, theta):
    qr1 = circuit.qregs[index_best]
    i=0
    for q in list_entang:
        circuit.cnot(qr1[i], q)
        theta[q]=theta[qr1[i]]
        i+=1


def applyMeasureOperator(circuit):
    for quantum_classical_registers in zip(circuit.qregs, circuit.cregs):
        for qubit_bit in zip(quantum_classical_registers[0], quantum_classical_registers[1]):
         circuit.measure(qubit_bit[0], qubit_bit[1])

def applyMultiHadamardOnList(circuit, list_qubit):
    for qubit in list_qubit:
        circuit.h(qubit)

def applyXOnList(circuit, list_qubit_X,dict_chr):
    for qubit in list_qubit_X:
        if dict_chr[qubit]=='1':
            circuit.x(qubit)


def computeLists(circuit, index_best, number_of_populations,num_genes):
    list_pop = [i for i in range(number_of_populations)]
    list_pop.remove(index_best)
    random.shuffle(list_pop)
    point=math.ceil(num_genes/(number_of_populations-1))
    k=0
    qr1 = circuit.qregs[index_best]
    list_qubit_mutation=[]
    list_qubit_gate=[]
    list_qubit_gate.extend([q for q in qr1])
    list_qubit_entang=[]
    #print(qr1)
    for ind in list_pop:
        qr2=circuit.qregs[ind]
        for i in range(k,min(k+point, num_genes)):
            #print("[", k, ",", k+point-1, "]")
            list_qubit_entang.append(qr2[i])
        list_qubit_mutation.extend([qr2[e] for e in range(0,num_genes) if e not in range(k,min(k+point, num_genes))])
        k=k+point
    list_qubit_gate.extend(list_qubit_mutation)
    return list_qubit_gate, list_qubit_entang, list_qubit_mutation, []

def updateListXElitismD(circuit, index_best, list_qubit_gate, list_qubit_X):
    for qb in circuit.qregs[index_best]:
        list_qubit_X.append(qb)
    for qb in list_qubit_X:
        list_qubit_gate.remove(qb)


def initializeLists(circuit):
    list_qubit_mutation=[]
    list_qubit_gate=[]
    list_qubit_entang=[]
    list_qubit_X = []
    for quantum_register in circuit.qregs:
        for qubit in quantum_register:
            list_qubit_gate.append(qubit)
    return list_qubit_gate, list_qubit_entang, list_qubit_mutation, list_qubit_X



class Parameters():
    def __init__(self, pop_size, max_gen, epsilon_init, prob_mut, elitism, num_shots=1, progressBar=False, verbose=True,draw_circuit=False, qobj_id=None):
        self.pop_size=pop_size
        self.max_gen=max_gen
        self.epsilon_init=epsilon_init
        self.prob_mut=prob_mut
        self.num_shots=num_shots
        self.elitism=elitism
        self.progressBar=progressBar
        self.verbose = verbose
        self.draw_circuit= draw_circuit
        self.qobj_id=qobj_id

    def __str__(self):
        return str(self.elitism)+"_eps_init_"+str(self.epsilon_init)+ "_prob_mut_"+str(self.prob_mut)

class ReinforcementParameters(Parameters):
    def __init__(self, pop_size,max_gen,epsilon_init, epsilon,prob_mut, elitism=ELITISM_R, num_shots=1, progressBar=False, verbose=True,draw_circuit=False,qobj_id=None):
        super().__init__(pop_size,max_gen,epsilon_init, prob_mut, elitism, num_shots, progressBar,verbose,draw_circuit,qobj_id)
        self.epsilon=epsilon

    def __str__(self):
        return str(self.elitism)+"_eps_init_"+str(self.epsilon_init)+ "_prob_mut_"+str(self.prob_mut)+"_eps_"+str(self.epsilon)
