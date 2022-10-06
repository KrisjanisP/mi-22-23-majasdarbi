#!/usr/bin/env python3

def summa(a:float, b:float)->float:
    return a+b
    

def atnemsana(a:float, b:float)->float:
    return a-b


def multiplikacija(a: float, b:float)->float:
    return a*b

def dalisana(a: float, b: float):
    assert b!=0
    return a/b

def eksponenta(arg1: float, arg2:float)->float:
    return arg1**arg2


if __name__ == "__main__":
    print("Testēt funkciju", __name__)
    assert summa(1, 2) == 3
    assert atnemsana(3, 1) == 2
    assert multiplikacija(1, 2) == 2
    assert dalisana(2, 2) == 1
    assert eksponenta(2, 2) == 4

# Šo daļu nedzēst!
# assert summa(1,2) == 3
# assert atnemsana(3,1) == 2
# assert multiplikacija(1,2) == 2
# assert dalisana(2,2) == 1
# assert eksponenta(2,2) == 4
