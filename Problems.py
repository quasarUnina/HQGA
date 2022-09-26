import math
import random
import intervalCode as cod
import hqga_utils as u

class Problem():
    def __init__(self):
        self.maxp = False


    def convertToReal(self, chr):
        #return value as the left point of the interval
        return cod.convertFromBinToFloat(chr, self.lower_bounds, self.upper_bounds, self.num_bit_code, self.dim)
        #return value as the middle point of the interval
        #return cod.convertFromBinToMiddleFloat(chr, self.lower_bounds, self.upper_bounds, self.num_bit_code, self.dim)

    def convertToBin(self,chr):
        return chr

    def isMaxProblem(self):
        return self.maxp

class RealProblem(Problem):
    def convert(self, chr):
        return self.convertToReal(chr)

    def computeFitness(self, chr):
        chr_real = self.convert(chr)
        return self.computeFitnessFromReal(chr_real)

    def getOptimumFitness(self):
        fitnesses=[]
        if self.dim==1:
            all_values=cod.getAllValues(self.lower_bounds[0], self.upper_bounds[0], self.num_bit_code)
            for val in all_values:
                chr_real=[val]
                fitnesses.append(self.computeFitnessFromReal(chr_real))
        if self.dim == 2:
            all_values = []
            for i in range(self.dim):
                all_values.append(cod.getAllValues(self.lower_bounds[i], self.upper_bounds[i], self.num_bit_code))
            for val1 in all_values[0]:
                for val2 in all_values[1]:
                    chr_real = [val1,val2]
                    fitnesses.append(self.computeFitnessFromReal(chr_real))
        if self.isMaxProblem():
            optimal_fit=max(fitnesses)
        else:
            optimal_fit=min(fitnesses)
        return optimal_fit

    def getOptimumValue(self):
        fitnesses=[]
        sols=[]
        if self.dim==1:
            all_values=cod.getAllValues(self.lower_bounds[0], self.upper_bounds[0], self.num_bit_code)
            best=all_values[0]
            best_f=self.computeFitnessFromReal([all_values[0]])
            for val in all_values:
                chr_real=[val]
                sols.append(val)
                fit=self.computeFitnessFromReal(chr_real)
                fitnesses.append(fit)
                if (u.isBetter(fit, best_f,self.isMaxProblem())):
                    best_f=fit
                    best=val
            best_sols=[]
            i=0
            for f in fitnesses:
                if f==best_f:
                    best_sols.append(sols[i])
                i+=1
        if self.dim == 2:
            all_values=[]
            for i in range(self.dim):
                all_values.append(cod.getAllValues(self.lower_bounds[i], self.upper_bounds[i], self.num_bit_code))
            best = [all_values[0][0],all_values[1][0]]
            best_f = self.computeFitnessFromReal(best)
            for val1 in all_values[0]:
                for val2 in all_values[1]:
                    chr_real = [val1, val2]
                    sols.append(chr_real)
                    fit = self.computeFitnessFromReal(chr_real)
                    fitnesses.append(fit)
                    if (u.isBetter(fit, best_f,self.isMaxProblem())):
                        best_f = fit
                        best = [val1, val2]
            best_sols = []
            i=0
            for f in fitnesses:
                if f==best_f:
                    best_sols.append(sols[i])
                i+=1
        return best_sols

    def getGrayOptimumValue(self):
        best_sols=self.getOptimumValue()
        sols=[]
        if self.dim==1:
            for sol in best_sols:
                sols.append(cod.convertFromFloatToBin(sol, self.lower_bounds[0], self.upper_bounds[0], self.num_bit_code))
        if self.dim == 2:
            for sol in best_sols:
                l=[]
                for i in range(self.dim):
                    l.append(cod.convertFromFloatToBin(sol[i], self.lower_bounds[i], self.upper_bounds[i], self.num_bit_code))
                sols.append(l)
        return sols

    def getFitnesses(self):
        fitnesses=[]
        sols=[ ]
        gray_sols=[]
        if self.dim==1:
            all_values=cod.getAllValues(self.lower_bounds[0], self.upper_bounds[0], self.num_bit_code)
            for val in all_values:
                chr_real=[val]
                sols.append(chr_real)
                gray_sols.append(cod.convertFromFloatToBin(chr_real, self.lower_bounds[0], self.upper_bounds[0], self.num_bit_code))
                fitnesses.append(self.computeFitnessFromReal(chr_real))
        if self.dim == 2:
            all_values = []
            for i in range(self.dim):
                all_values.append(cod.getAllValues(self.lower_bounds[i], self.upper_bounds[i], self.num_bit_code))
            for val1 in all_values[0]:
                for val2 in all_values[1]:
                    chr_real = [val1,val2]
                    sols.append(chr_real)
                    l_tmp=[]
                    for i in range(self.dim):
                        l_tmp.append(cod.convertFromFloatToBin(chr_real[i], self.lower_bounds[i], self.upper_bounds[i],self.num_bit_code))
                    gray_sols.append(l_tmp)
                    fitnesses.append(self.computeFitnessFromReal(chr_real))
        return sols, gray_sols, fitnesses





class RastriginProblem(RealProblem):
    def __init__(self, dim = 2, lower_bounds = [-5.12,-5.12], upper_bounds = [5.12,5.12], A=10, num_bit_code = 4):
        self.name="Rastrigin"
        self.A=A
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        #compute fitness value
        res=self.A*self.dim+sum([x**2-10*math.cos(2*math.pi*x) for x in chr_real])
        #print("fitness value ",res)
        return res

class SphereProblem(RealProblem):
    def __init__(self,dim = 2, lower_bounds = [-5.12,-5.12], upper_bounds = [5.12,5.12], num_bit_code = 4):
        self.name="Sphere"
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        res = sum([i ** 2 for i in chr_real])
        #print("fitness value ", res)
        return res



class oneMaxProblem(Problem):
    def __init__(self, dim=5):
        self.name="OneMax"
        self.maxp = True
        self.num_bit_code=1
        self.dim=dim

    def convert(self, chr):
        return self.convertToBin(chr)

    def computeFitness(self, chr):
        return chr.count("1")

    def getOptimumFitness(self):
        return self.dim

class BealeProblem(RealProblem):
    def __init__(self,dim = 2, lower_bounds = [-4.5,-4.5], upper_bounds = [4.5,4.5], num_bit_code = 4):
        self.name="Beale"
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x1=chr_real[0]
        x2=chr_real[1]
        #compute fitness value
        res=(1.5 -x1+x1*x2)**2 + (2.25 -x1 + x1*x2**2)**2 + (2.625 -x1 + x1*x2**3)**2
        #print("fitness value ",res)
        return res


class RosenbrockProblem(RealProblem):
    def __init__(self, dim = 2, lower_bounds = [-2.048,-2.048], upper_bounds = [2.048, 2.048], num_bit_code = 4):
        self.name="Rosenbrock"
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        #compute fitness value
        x1 = chr_real[0]
        x2 = chr_real[1]
        res=100*(x2 - x1**2)**2+(1-x1)**2
        #print("fitness value ",res)
        return res

class StepProblem(RealProblem):
    def __init__(self, dim = 2, lower_bounds = [-5.12,-5.12], upper_bounds = [5.12,5.12], num_bit_code = 4):
        self.name="Step"
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        res = sum([int(i) for i in chr_real])
        #print("fitness value ", res)
        return res

class QuarticProblem(RealProblem):
    def __init__(self, dim = 2, lower_bounds = [-1.28,-1.28], upper_bounds = [1.28, 1.28], num_bit_code = 4):
        self.name="Quartic"
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        res = sum([(i+1)*chr_real[i]**4+random.gauss(0,1) for i in range(len(chr_real))])
        #print("fitness value ", res)
        return res

class SchwefelProblem(RealProblem):
    def __init__(self,dim = 2, lower_bounds = [-500,-500], upper_bounds = [500,500], V = 418.9829, num_bit_code = 4):
        self.name="Schwefel"
        self.V = V
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        res = self.dim*self.V + sum([-x*math.sin(math.sqrt(abs(x))) for x in chr_real])
        #print("fitness value ", res)
        return res


class GriewangkProblem(RealProblem):
    def __init__(self, dim = 2, lower_bounds = [-600,-600], upper_bounds = [600,600],num_bit_code = 4):
        self.name="Griewangk"
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        res =1 + sum([x**2/4000 for x in chr_real])- math.prod([math.cos(chr_real[i]/math.sqrt(i+1)) for i in range(len(chr_real))])
        #print("fitness value ", res)
        return res

class AckleyProblem(RealProblem):
    def __init__(self, dim = 2, lower_bounds = [-32.768, -32.768], upper_bounds = [32.768, 32.768], a=20, b=0.2, c=2*math.pi,num_bit_code = 4):
        self.name="Ackley"
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.a=a
        self.b=b
        self.c=c

    def computeFitnessFromReal(self, chr_real):
        firstSum=sum([x**2 for x in chr_real])
        secondSum=sum([math.cos(self.c*x) for x in chr_real])
        res = -self.a*math.exp(-self.b*math.sqrt(firstSum/self.dim)) - math.exp(secondSum/self.dim) + self.a + math.exp(1)
        #print("fitness value ", res)
        return res

class BohachevskyProblem(RealProblem):
    def __init__(self, lower_bounds = [-100,-100], upper_bounds = [100,100], num_bit_code = 4):
        self.name="Bohachevsky"
        self.maxp=False
        self.dim = 2
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x1=chr_real[0]
        x2=chr_real[1]
        res = x1**2 + 2*x2**2 -0.3*math.cos(3*math.pi*x1) -0.4*math.cos(4*math.pi*x2)+0.7
        #print("fitness value ", res)
        return res

class BirdProblem(RealProblem):
    def __init__(self, lower_bounds = [-2*math.pi,-2*math.pi], upper_bounds = [2*math.pi,2*math.pi], num_bit_code = 4):
        self.name="Bird"
        self.maxp=False
        self.dim = 2
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        #print("cromosome ", chr_real)
        x=chr_real[0]
        y=chr_real[1]
        res = math.sin(x)*math.exp((1-math.cos(y))**2) + math.cos(y)*math.exp((1-math.sin(x))**2) + (x-y)**2
        #print("fitness value ", res)
        return res

class BoothProblem(RealProblem):
    def __init__(self, lower_bounds = [-10,-10], upper_bounds = [10,10], num_bit_code = 4):
        self.name="Booth"
        self.maxp=False
        self.dim = 2
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        y=chr_real[1]
        res = (x+2*y-7)**2 + (2*x + y - 5)**2
        #print("fitness value ", res)
        return res

class GoldeinsteinProblem(RealProblem):
    def __init__(self, lower_bounds = [-2,-2], upper_bounds = [2,2], num_bit_code = 4):
        self.name="Goldeinstein"
        self.maxp=False
        self.dim = 2
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        X=chr_real[0]
        Y=chr_real[1]
        res = (1 + ((X + Y + 1)**2) * (19 - (14 * X) + (3 * (X**2)) - 14*Y + (6 *X*Y) + (3 * (Y**2)))) *(30 + ((2 * X - 3 * Y)**2) * (18 - 32 * X + 12 * (X**2) + 48 * Y - (36* X*Y) + (27 * (Y**2))) )
        #print("fitness value ", res)
        return res

class HolderTableProblem(RealProblem):
    def __init__(self, lower_bounds = [-10,-10], upper_bounds = [10,10], num_bit_code = 4):
        self.name="HolderTable"
        self.maxp=False
        self.dim = 2
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        X=chr_real[0]
        Y=chr_real[1]
        res = -(abs(math.sin(X)*math.cos(Y)*math.exp(abs(1-(math.sqrt(X**2+Y**2)/math.pi)))))
        #print("fitness value ", res)
        return res

class GramacyProblem(RealProblem):
    def __init__(self, lower_bounds = [0.5], upper_bounds = [2.5], num_bit_code = 8):
        self.name="Gramacy"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        X=chr_real[0]
        res = (math.sin(10*math.pi*X)/2*X) +(X-1)**4
        #print("fitness value ", res)
        return res

class ForresterProblem(RealProblem):
    def __init__(self, lower_bounds = [0], upper_bounds = [1], num_bit_code = 8):
        self.name="Forrester"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = (6*x-2)**2 * math.sin(12*x - 4)
        #print("fitness value ", res)
        return res


class PeriodicProblem(RealProblem):
    def __init__(self, dim = 2, lower_bounds = [-2, -2], upper_bounds = [2,2], num_bit_code = 4):
        self.name="Periodic"
        self.maxp=False
        self.dim = dim
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds


    def computeFitnessFromReal(self, chr_real):
        firstSum=sum([math.sin(x)**2 for x in chr_real])
        secondSum=sum([x**2 for x in chr_real])
        res = 1 + firstSum - 0.1 * math.e**(-secondSum)
        #print("fitness value ", res)
        return res



class Problem02(RealProblem):
    def __init__(self, lower_bounds = [2.7], upper_bounds = [7.5], num_bit_code = 8):
        self.name="Problem02"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = math.sin(x) + math.sin(10/3*x)
        #print("fitness value ", res)
        return res


class Problem03(RealProblem):
    def __init__(self, lower_bounds = [-10], upper_bounds = [10], num_bit_code = 8):
        self.name="Problem03"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = -sum([k*math.sin((k+1)*x + k) for k in range(1,7)])
        #print("fitness value ", res)
        return res

class Problem04(RealProblem):
    def __init__(self, lower_bounds = [1.9], upper_bounds = [3.9], num_bit_code = 8):
        self.name="Problem04"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = -(16*x**2 - 24*x + 5)* math.e**(-x)
        #print("fitness value ", res)
        return res

class Problem05(RealProblem):
    def __init__(self, lower_bounds= [0.0], upper_bounds = [1.2], num_bit_code = 8):
        self.name="Problem05"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = -(1.4 -3*x)*math.sin(18*x)
        #print("fitness value ", res)
        return res

class Problem06(RealProblem):
    def __init__(self, lower_bounds = [-10], upper_bounds = [10], num_bit_code = 8):
        self.name="Problem06"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = -(x + math.sin(x))*math.e**(-x**2)
        #print("fitness value ", res)
        return res

class Problem07(RealProblem):
    def __init__(self, lower_bounds = [2.7], upper_bounds = [7.5], num_bit_code = 8):
        self.name="Problem07"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = math.sin(x) + math.sin(10/3*x) + math.log(x) - 0.84*x + 3
        #print("fitness value ", res)
        return res

class Problem08(RealProblem):
    def __init__(self, lower_bounds = [-10], upper_bounds = [10], num_bit_code = 8):
        self.name="Problem08"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = -sum([k*math.cos((k+1)*x + k) for k in range(1,7)])
        #print("fitness value ", res)
        return res

class Problem09(RealProblem):
    def __init__(self, lower_bounds = [3.1], upper_bounds = [20.4], num_bit_code = 8):
        self.name="Problem09"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = math.sin(x) + math.sin(2/3*x)
        #print("fitness value ", res)
        return res

class Problem10(RealProblem):
    def __init__(self, lower_bounds = [0], upper_bounds = [10], num_bit_code = 8):
        self.name="Problem10"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = - x * math.sin(x)
        #print("fitness value ", res)
        return res

class Problem13(RealProblem):
    def __init__(self, lower_bounds = [0.001], upper_bounds= [0.99], num_bit_code = 8):
        self.name="Problem13"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = -x**(2/3) - (1 - x**2)**(1/3)
        #print("fitness value ", res)
        return res

class Problem14(RealProblem):
    def __init__(self, lower_bounds = [0], upper_bounds = [4], num_bit_code = 8):
        self.name="Problem14"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = -math.e**(-x)*math.sin(2*math.pi*x)
        #print("fitness value ", res)
        return res

class Problem15(RealProblem):
    def __init__(self, lower_bounds = [-5], upper_bounds = [5], num_bit_code = 8):
        self.name="Problem15"
        self.maxp=False
        self.dim = 1
        self.num_bit_code = num_bit_code
        self.lower_bounds= lower_bounds
        self.upper_bounds = upper_bounds

    def computeFitnessFromReal(self, chr_real):
        x=chr_real[0]
        res = (x**2 -5*x + 6)/(x**2 + 1)
        #print("fitness value ", res)
        return res


class ProblemF1(Problem02):
    def __init__(self,lower_bounds = [2.7], upper_bounds = [7.5], num_bit_code=8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F1"

class ProblemF2(Problem04):
    def __init__(self, lower_bounds = [1.9], upper_bounds = [3.9], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F2"

class ProblemF3(Problem05):
    def __init__(self, lower_bounds= [0.0], upper_bounds = [1.2], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F3"

class ProblemF4(Problem06):
    def __init__(self, lower_bounds = [-10], upper_bounds = [10], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F4"

class ProblemF5(Problem07):
    def __init__(self, lower_bounds = [2.7], upper_bounds = [7.5], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F5"

class ProblemF6(Problem09):
    def __init__(self, lower_bounds = [3.1], upper_bounds = [20.4], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F6"

class ProblemF7(Problem10):
    def __init__(self, lower_bounds = [0], upper_bounds = [10], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F7"

class ProblemF8(Problem13):
    def __init__(self, lower_bounds = [0.001], upper_bounds= [0.99], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F8"

class ProblemF9(Problem14):
    def __init__(self, lower_bounds = [0], upper_bounds = [4], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F9"

class ProblemF10(Problem15):
    def __init__(self, lower_bounds = [-5], upper_bounds = [5], num_bit_code = 8):
        super().__init__(lower_bounds, upper_bounds , num_bit_code)
        self.name="F10"
     
