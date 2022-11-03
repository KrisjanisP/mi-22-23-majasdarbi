#!/usr/bin/env python3
'''
Python 6 nodarbības mājasdarbs Nr.1

Uzdevums: aizpildīt vietas ar atzīmi TODO
'''
import numpy as np

# izveidot numpy random 3x3 matricu
arr = np.random.rand(3,3)

# izveidot citu numpy matricu, kas ir inversā matrica no pirmās matricas
arr_inv = np.linalg.inv(arr) 

# sareizināt abas matricas un noapaļot līdz integer precizitātei
reizinājums = np.rint(np.dot(arr,arr_inv)).astype(int)

# Izveidot trešo matricu, kas ir 3x3 identitātes matrica 
identitates_matrica = np.eye(3,3)

# Pārbaudīt vai identitates_matrica ir vienāda ar reizinājumu!
vienads = np.array_equal(reizinājums, identitates_matrica)

# Lai pārbaudītu iznākumu, atkomentēt nākamo rindu
assert vienads == True

print(arr)
print(arr_inv)
print(reizinājums)
print(identitates_matrica)
print(vienads)