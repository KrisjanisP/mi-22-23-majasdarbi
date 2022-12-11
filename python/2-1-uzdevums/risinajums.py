#!/usr/bin/env python3

## 1. uzdevums 
''' Funkcija summe divus skaitļus.

    input: divi float argumenti 
    
    output: summas vērtība
            (return 0 var izdzēst)
'''
def summa(a:float, b:float)->float:
    return a+b
    

## 2. uzdevums
''' Funkcija, kas atņem divus skaitļus.

    input: divi float argumenti 
    
    output: atņemšanas vērtība
'''
def atnemsana(a:float, b:float)->float:
    return a-b


## 3. uzdevums

'''
    Funkcija, kas multiplicē divus skaitļus.

    input: divi float argumenti 
    
    output: multiplicēšanas vērtība
'''
def multiplikacija(a: float, b:float)->float:
    return a*b

## 4. uzdevums
''' Funkcija, kas dala divus skaitļus. Ja nulle, tad izbeidz programmu

    input: divi float argumenti 
    
    output: dalījuma vērtība
'''
def dalisana(a: float, b: float):
    assert b!=0, "b nevar būt 0"
    return a/b

## 5. uzdevums
''' Funkcija, kas eksponē arg1 pakāpē arg2.

    input: divi float argumenti 
    
    output: eksponenta vērtība
'''
def eksponenta(arg1: float, arg2:float)->float:
    return arg1**arg2


def parbaudit_ievadi(ievade):
    ''' Funkcija, kas paņem pirmo ievadi. Atkarībā no ievades, prasīt lietotājam vajadzīgos skaitļus
            ievades skaitļus pārbaudīt

        input: arguments ar nezināmu datu tipu

    '''

    #  ja ir tad atgriež float vērtību, ja ne tad pārtrauc programmu
    # 
    ''' Funkcija, kas pārbauda vai arguments ir numerisks
        Ja nav numerisks, tad pārtrauc programmu.

        input: arguments ar nezināmu datu tipu
        
        output: ievades vērtība kā float datu tips
                (return 0 var izdzēst)
    '''
    def check_float(x: float)->float:
        assert isinstance(x,float) or isinstance(x,int)
        return x
        
    skaitlis_1 = input("Ievadīt pirmo vērtību summēšanai:")
    skaitlis_1 = check_float( skaitlis_1 )
    skaitlis_2 = input("Ievadīt otro vērtību summēšanai:")
    skaitlis_2 = check_float( skaitlis_2 )

    if ievade == '+':
        print("rezultāts: ", summa(skaitlis_1,skaitlis_2))
    elif ievade == '-':
        print("rezultāts: ", atnemsana(skaitlis_1,skaitlis_2))
    elif ievade == '*':
        print("rezultāts: ", multiplikacija(skaitlis_1,skaitlis_2))
    elif ievade == '/':
        print("rezultāts: ", dalisana(skaitlis_1,skaitlis_2))
    elif ievade == 'eksp':
        print("rezultāts: ", eksponenta(skaitlis_1,skaitlis_2))
    else:
        print("operācija nav atrasta. Izvēlēties  [+ , - , * , / , eksp  ]")
        exit()


## Šeit sākās maģija
ievade = input("Vēlamo darbību, atļauts [+ , - , * , / , eksp  ]: ")

parbaudit_ievadi(ievade)

## Šo daļu nedzēst!
#assert summa(1,2) == 3
#assert atnemsana(3,1) == 2
#assert multiplikacija(1,2) == 2
#assert dalisana(2,2) == 1
#assert eksponenta(2,2) == 4