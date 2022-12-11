import random
###########################
########### setup
###########################
# Ideja: https://www.youtube.com/watch?v=pKO9UjSeLew&ab_channel=JomaTech
n=7
s = set(range(1,n ))
l=[random.randint(1, n-1) for x in range(n)]
print(l)


###########################
########### atrodi dubultos
###########################
# Atrodi visus dubultos skaitļus un ievieto tos doubles listē
# l ir 7 gara liste ar numuriem 1-6 => būs vienmēr situacija
# kad būs dubulti skaitļi
# s ir set ar numuriem 1-6
doubles = set()
found = set()
for x in l:
    if x in found:
        doubles.add(x)
    found.add(x)
print(doubles)


    