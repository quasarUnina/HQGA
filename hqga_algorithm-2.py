from qiskit import execute
import math
import copy
from tqdm import tqdm

import time
import intervalCode as cod
import utils as u
import hqga_utils
import matplotlib.pyplot as plt

from qiskit.compiler import transpile
from qiskit import BasicAer
from qiskit.visualization import plot_bloch_multivector


from qiskit.providers.jobstatus import JOB_FINAL_STATES


def updateThetaReinforcement(dict_chr, theta, epsilon,index_best, num_genes):
    i=0
    for key in dict_chr.keys():
       if i in range(index_best*num_genes,index_best*num_genes+num_genes):
          if dict_chr[key]=='1':
               if  theta[key]>0 and theta[key]<math.pi:
                    theta[key]=theta[key]+epsilon
               else:
                   theta[key]=theta[key]-epsilon
          else:
                if theta[key]>0  and  theta[key]<math.pi:
                    theta[key] = theta[key] - epsilon
                else:
                    theta[key] = theta[key] + epsilon
       i+=1

def updateThetaReinforcementWithinRange(dict_chr, theta, epsilon,index_best, num_genes):
    i=0
    for key in dict_chr.keys():
        if i in range(index_best*num_genes,index_best*num_genes+num_genes):
            if dict_chr[key]=='1':
                theta[key]=theta[key]+epsilon
                if theta[key] > math.pi/2:
                    theta[key] =  math.pi/2
            else:
                theta[key] = theta[key] - epsilon
                if theta[key] < -math.pi/2:
                    theta[key] = - math.pi/2
        i+=1

def resetThetaReinforcement(dict_chr, theta, old_theta, old_index_best, num_genes):
    i=0
    for key in dict_chr.keys():
        if i in range(old_index_best*num_genes,old_index_best*num_genes+num_genes):
            theta[key]=old_theta[key]
        i+=1

#def updateTheta(dict_chr, theta, index_best, num_genes):
#    i=0
#    for key in dict_chr.keys():
#        if i in range(index_best*num_genes,index_best*num_genes+num_genes):
#            if dict_chr[key]=='1':
#                theta[key]=math.pi
#            else:
#                theta[key]=0
#        i+=1

def runQGA(device_features,circuit, params,problem):
    chromosome_evolution=[]
    bests =[]
    gen=0
    theta = hqga_utils.initializeTheta(circuit,params.epsilon_init)
    #theta=hqga_utils.initializeThetaWithoutHadamard(circuit, params.epsilon_init)
    #theta=qga_utils.initializeThetaWithoutHadamardWithoutParameter(circuit)
    #u.save_obj(theta, "dict_theta_"+str(problem.dim*problem.num_bit_code))
    #theta = u.load_obj("dict_theta")
    gBest = hqga_utils.globalBest()
    list_qubit_gate, list_qubit_entang, list_qubit_mutation, list_qubit_X=hqga_utils.initializeLists(circuit)
    dict_chr=[]
    flag_index_best=False
    #while gen<=max_gen:
    l_gen=range(params.max_gen+1)
    if params.progressBar:
        l_gen= tqdm(range(params.max_gen+1), desc="Generations")
    for gen in l_gen:
        #print("\n########## generation #########", gen)
        hqga_utils.applyMultiHadamardOnList(circuit, list_qubit_gate)
        hqga_utils.applyMultiRotationOnList(circuit, theta, list_qubit_gate)
        hqga_utils.applyXOnList(circuit, list_qubit_X,dict_chr )
        if gen!=0:
            circuit.barrier()
            hqga_utils.applyEntanglementOnList(circuit, index_best, list_qubit_entang, theta)
            circuit.barrier()
            #hqga_utils.applyMutationOnList(circuit, params.prob_mut, math.pi, list_qubit_mutation, theta)
            hqga_utils.applyMutationOnListWithinRange(circuit, params.prob_mut, list_qubit_mutation, theta)
            #qga_utils.applyMutationWithoutParameterOnList(circuit, list_qubit_mutation, theta)
            #qga_utils.applyQuantumMutationOnList(circuit, list_qubit_mutation, theta, epsilon_mut)
        circuit.barrier()

        #backendBlochSphere = BasicAer.get_backend('statevector_simulator')
        #backend = BasicAer.get_backend('qasm_simulator') #per simulazioni
        #circuit_w=copy.deepcopy(circuit)
        #job = execute(circuit_w, backendBlochSphere).result()
        #print("statevector ", len(job.get_statevector(circuit_w)))
        #ax = plot_bloch_multivector(job.get_statevector(circuit_w))
        #ax.savefig("spherebloch"+str(gen)+".png")
        # u.save_obj(ax, "spherebloch"+str(gen))
        #plt.show()

        hqga_utils.applyMeasureOperator(circuit)
        # Draw the circuit
        if params.draw_circuit:
            #print(circuit.draw(output="text", fold=300))
            #print(circuit.draw(output="mpl", fold=300, filename="circuit_"+str(gen)+".png"))
            display(circuit.draw(output="mpl", scale=0.8, fold=30))
            #print(circuit.draw(output="latex_source", fold=300))
            #print(circuit.qasm())

        print("depth",circuit.depth())
        # Execute the circuit on the qasm simulator
        #save object circuit
        #if gen==0:
            #u.save_obj(circuit, "circuit0")
        #experiments = transpile(circuit,backend=simulator)
        #job = execute(experiments, simulator, shots=params.num_shots)

        while True:
            try:
                if device_features.real:
                    job = execute(circuit, device_features.device, shots=params.num_shots, qobj_id=str(params.qobj_id)+str(gen))
                else:
                    job = execute(circuit, device_features.device, noise_model=device_features.noise_model,
                               coupling_map=device_features.coupling_map,
                               basis_gates=device_features.basis_gates, shots=params.num_shots, qobj_id=str(params.qobj_id)+str(gen))
                while job.status() not in JOB_FINAL_STATES:
                    #print(job.status())
                    pass
                # Grab results from the job
                result = job.result()
                #print(result)
                break
            except Exception as e:
                print(e)

        # Returns counts
        counts = result.get_counts(circuit)
        #print("\nCounts:",counts)
        #print("len counts ", len(counts))
        #print(hqga_utils.getMaxProbKey(counts))
        # plot_histogram(counts).show()
        #print("Average fitness values for each population",hqga_utils.averageFitnessonPopulation(counts, number_of_populations, gene_per_chromosome, params.num_shots, problem.computeFitness,gen,False))
        #compute fitness evaluation
        classical_chromosomes=hqga_utils.fromQtoC(hqga_utils.getMaxProbKey(counts))
        if params.verbose:
            print("\nChromosomes", classical_chromosomes)
            #for g in range(params.pop_size):
            #    new_lower_bounds, new_upper_bounds = cod.convertFromBinToInterval(classical_chromosomes[g], problem.lower_bounds,
            #                                                                  problem.upper_bounds,
            #                                                                  problem.num_bit_code, problem.dim)
            #    print("Interval chromosome ", str(g+1) , new_lower_bounds, new_upper_bounds )
        l_sup=[]
        for c in classical_chromosomes:
            l_sup.append(problem.convert(c))
        chromosome_evolution.append(l_sup)
        if params.verbose:
            print("\nReal Chromosomes", l_sup)
        fitnesses=hqga_utils.computeFitnesses(classical_chromosomes, problem.computeFitness)
        if params.verbose:
            print("fitness values ", fitnesses)
        if gen!=0:
            previous_best=index_best
        best_fitness, index_best=hqga_utils.computeBest(fitnesses, problem.isMaxProblem())
        #print("best fitness", best_fitness, " index best ", index_best)
        if gen == 0:
            gBest.chr = classical_chromosomes[index_best]
            gBest.fitness = best_fitness
            #gBest.theta=theta_best
            gBest.gen = params.pop_size
        else:
            flag_index_best = previous_best != index_best
            if hqga_utils.isBetter(best_fitness, gBest.fitness, problem.isMaxProblem()):
                gBest.chr = classical_chromosomes[index_best]
                gBest.fitness = best_fitness
                #gBest.theta = theta_best
                gBest.gen = (gen+1)*params.pop_size
        bests.append([problem.convert(gBest.chr), gBest.fitness, gBest.chr])


        list_qubit_gate, list_qubit_entang,list_qubit_mutation,list_qubit_X=hqga_utils.computeLists(circuit, index_best, params.pop_size, problem.dim*problem.num_bit_code)
        #print("theta before updating",theta.values())
        if params.elitism is not hqga_utils.ELITISM_Q:
            #update
            dict_chr = hqga_utils.create_dict_chr(circuit, classical_chromosomes)
            #print(dict_chr.values())
            #print("theta prima",theta.values())
            if params.elitism is hqga_utils.ELITISM_R:
                if flag_index_best:
                    resetThetaReinforcement(dict_chr, theta, old_theta, previous_best, problem.dim*problem.num_bit_code)
                old_theta=copy.deepcopy(theta)
                updateThetaReinforcementWithinRange(dict_chr, theta, params.epsilon, index_best, problem.dim*problem.num_bit_code)
            elif params.elitism is hqga_utils.ELITISM_D:
                #used with rotation
                #updateTheta(dict_chr, theta, index_best, gene_per_chromosome)
                hqga_utils.updateListXElitismD(circuit, index_best, list_qubit_gate, list_qubit_X)
            else:
                raise Exception("Value for elitism is not valid.")
        #print("theta after updating", theta.values())
        #print(theta.values())

        hqga_utils.resetCircuit(circuit)
        #gen+=1

    gBest.display()
    print("The total number of fitness evaluations is: ", params.pop_size*(params.max_gen+1))
    return gBest, chromosome_evolution,bests

