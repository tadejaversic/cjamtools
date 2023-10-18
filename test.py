import numpy as np

bla = np.arange(3)
prod = 1
for i in bla:
    prod*= (i+1)
    print(i)
print("Final product is ", prod)
