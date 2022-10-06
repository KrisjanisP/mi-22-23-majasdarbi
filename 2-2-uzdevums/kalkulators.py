import sys
from calc import *


def parbaudit_ievadi():

    def check_float(x):
        if not x.isnumeric():
            exit()
        return float(x)

    if sys.argv[2] == '+':
        print(summa(check_float(sys.argv[1]), check_float(sys.argv[3])))
    elif sys.argv[2] == '-':
        print(atnemsana(check_float(sys.argv[1]), check_float(sys.argv[3])))
    elif sys.argv[2] == '*':
        print(multiplikacija(check_float(sys.argv[1]), check_float(sys.argv[3])))
    elif sys.argv[2] == '/':
        print(dalisana(check_float(sys.argv[1]), check_float(sys.argv[3])))
    elif sys.argv[2] == '**':
        print(eksponenta(check_float(sys.argv[1]), check_float(sys.argv[3])))
    else:
        exit()


print(sys.argv)

parbaudit_ievadi()
