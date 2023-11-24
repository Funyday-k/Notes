import math

def pi_Leibniz_calc(n):
    pp = 0.0
    i=0.0
    while(i<n):
        pp += pow(-1,i)/(2 * i + 1)
        i += 1.0
    pp *= 4 
    print(pp)

def pi_Euler_calc(n):
    pp = 0.0
    i=1
    while(i<n):
        pp += 1/(i ** 2)
        i += 1
    pp *= 6
    pp = math.sqrt(pp)
    print(pp)

def pi_Ramanujan_calc(n):
    pp = 0.0
    i=0
    while(i<n):
        pp += math.factorial(4*i)*(1103 + 26390 * i)/((math.factorial(i) ** 4 )*(396 ** (4*i)))
        i += 1
    pp *= math.sqrt(8)/(99 ** 2)
    pp = 1/pp
    print(pp)
