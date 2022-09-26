from sympy.combinatorics.graycode import gray_to_bin
from sympy.combinatorics.graycode import bin_to_gray


#lower_bound and upper_bound for each gene
#num_bit_code: bits to represent each gene
# dim: the dimension of the chromosome
#chr: the binary chromosome in gray code
def convertFromBinToFloat(chr, lower_bounds, upper_bounds,num_bit_code,dim):
    genes = []
    chr_real = []
    step = [(x-y)/(2 ** num_bit_code - 1) for x,y in zip(upper_bounds,lower_bounds)]
    k = 0
    #print(dim)
    #print(chr)
    for i in range(dim):
        var = chr[k:k + num_bit_code]
        #print("cromosome ", var)
        #print(gray_to_bin(var))
        #print(int(gray_to_bin(var), 2))
        genes.append(int(gray_to_bin(var), 2))
        # bcd_num=int(chr, 2)
        # num=bcd.bcd_to_int(bcd_num)
        gene_real = lower_bounds[i] + genes[i] * step[i]
        chr_real.append(gene_real)
        k = k + num_bit_code
    return chr_real


#lower_bound and upper_bound for each gene
#num_bit_code: bits to represent each gene
# dim: the dimension of the chromosome
#chr: the binary chromosome in gray code
def convertFromBinToMiddleFloat(chr, lower_bounds, upper_bounds,num_bit_code,dim):
    genes = []
    chr_real = []
    step = [(x-y)/(2 ** num_bit_code - 1) for x,y in zip(upper_bounds,lower_bounds)]
    k = 0
    #print(dim)
    #print(chr)
    for i in range(dim):
        var = chr[k:k + num_bit_code]
        #print("cromosome ", var)
        #print(gray_to_bin(var))
        #print(int(gray_to_bin(var), 2))
        genes.append(int(gray_to_bin(var), 2))
        # bcd_num=int(chr, 2)
        # num=bcd.bcd_to_int(bcd_num)
        low_b = lower_bounds[i] + genes[i] * step[i]
        upp_b= low_b + step[i]
        gene_real=(low_b +upp_b)/2
        chr_real.append(gene_real)
        k = k + num_bit_code
    return chr_real

def convertFromBinToInterval(chr, lower_bounds_input, upper_bounds_input,num_bit_code,dim):
    genes = []
    lower_bounds = []
    upper_bounds = []
    step = [(x-y)/(2 ** num_bit_code - 1) for x,y in zip(upper_bounds_input,lower_bounds_input)]
    k = 0
    #print(dim)
    #print(chr)
    for i in range(dim):
        var = chr[k:k + num_bit_code]
        #print("cromosome ", var)
        #print(gray_to_bin(var))
        #print(int(gray_to_bin(var), 2))
        genes.append(int(gray_to_bin(var), 2))

        # bcd_num=int(chr, 2)
        # num=bcd.bcd_to_int(bcd_num)
        gene_real = lower_bounds_input[i] + genes[i] * step[i]
        #print(gene_real)
        lower_bounds.append(gene_real)
        upper_bounds.append(gene_real+step[i])
        k = k + num_bit_code
    return lower_bounds,upper_bounds

def convertFromFloatToBin(value, lower_bound, upper_bound, num_bit_code):
    step = (upper_bound - lower_bound) / (2 ** num_bit_code - 1)
    index= int(round((value - lower_bound)/step,1))
    binary="{0:b}".format(index)
    gray_value=bin_to_gray(binary)
    if len(gray_value) <num_bit_code:
        gray_value=(num_bit_code-len(gray_value))*'0'+gray_value
    return gray_value



#all values for a gene within [lower_bound,upper_bound] and encoded with num_bit_code bits
def getAllValues(lower_bound, upper_bound,num_bit_code):
    list_values = []
    step = (upper_bound - lower_bound) / (2 ** num_bit_code - 1)
    #print("step",step)
    for i in range(2 ** num_bit_code):
        gene_real = lower_bound + i * step
        list_values.append(gene_real)
    return list_values

if __name__ == "__main__":
    gray='000010110000'
    float_v=convertFromBinToFloat(gray, [-1,-1,-1], [1,1,1],4,3)
    print(float_v)
    print(convertFromFloatToBin(-0.7333333333333334, -1, 1,4)+'')
    print(getAllValues(-1,1,4))
    for k in getAllValues(-1,1,4):
        print(convertFromFloatToBin(k, -1, 1, 4) + '')
