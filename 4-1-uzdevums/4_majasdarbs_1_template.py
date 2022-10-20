#!/usr/bin/env python3
'''
Python  mājasdarbs Nr.4-1

Uzdevums: aizpildīt vietas ar atzīmi TODO
'''


## Jānis un Andris savu draugu telefonnumurus ir saglabājuši vārdnīcā.
Janis = {"Andris" : "041123451", "Martins" : "041123457", "Santa" : "041123458", "Kristine" : "041123459"}
Andris = {"Janis" : "041123456", "Martins" : "041123457", "Santa" : "041123458", "Kristina" : "041123450"}

## Uzdevums 1 - Izvadīt to draugu vārdus, kuru numuri ir vienādi
numri = set()
for key in set(Janis.keys()) & set(Andris.keys()):
    if Janis[key] == Andris[key]:
        numri.add(key)

assert numri == {'Santa', 'Martins'}
print("vienādi numuri ir ", numri)

## Uzdevums 2 - Izvadīt to draugu vārdus, kuru numuri ir satopami Jāņa vārdnīcā, be nav Andra
numri = set()
for key in set(Janis.keys()) - set(Andris.keys()):
    numri.add(key) 

assert numri == {'Andris', 'Kristine'}
print("Jāņa vārdnīcā ir unikāli sekojošie draugi: ", numri)

## Uzdevums 3 - Izvadīt to draugu vārdus, kuru numuri ir sastopami TIKAI Jāņa vārdnīcā vai TIKAI Andra vārdnīcā (unikāli draugi)
numri = set() 
for key in set(Janis.keys()) ^ set(Andris.keys()):
    numri.add(key) 
assert numri == {'Janis', 'Kristina', 'Kristine', 'Andris'}
print("Unikāli draugi ir ", numri)

## Uzdevums 4 - Martins paziņo Jānim un Andrim ka maina telefona numuru. Modificēt abu vārdnīcas
Janis["Martins"] = "041123400"
Andris["Martins"] = "041123400"

assert Janis == {'Andris': '041123451', 'Martins': '041123400', 'Santa': '041123458', 'Kristine': '041123459'}
assert Andris == {'Janis': '041123456', 'Martins': '041123400', 'Santa': '041123458', 'Kristina': '041123450'}
print("Jāņa Vārdnīca ", Janis)
print("Andra Vārdnīca ", Andris)

## Uzdevums 5 - Andris iepazīstas ar Tomu, kura telefonnumurs ir 041999888. Ievietot Andra vārdnicā jauno draugu.
Andris["Toms"] = "041999888"

assert Andris == {'Janis': '041123456', 'Martins': '041123400', 'Santa': '041123458', 'Kristina': '041123450', 'Toms': '041999888'}
print("Andra Vārdnīca ", Andris)

## Uzdevums 6 - Jānis sastrīdās ar Santu un grib izdzēst viņas numuru.
Janis.pop("Santa")
assert Janis == {'Andris': '041123451', 'Martins': '041123400', 'Kristine': '041123459'}
print("Jāņa Vārdnīca ", Janis)