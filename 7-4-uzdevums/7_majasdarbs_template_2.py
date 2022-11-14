#!/usr/bin/env python3
'''
Python 7 mājasdarbs Nr.2

Uzdevums: aizpildīt vietas ar atzīmi TODO

'''

from IntroOpenCV import IntroOpenCV
from TopWords import TopWords

class Majasdarbs(TopWords, IntroOpenCV):
    '''

    '''



if __name__ == "__main__":
    obj = Majasdarbs()

    obj.set_picture("python.jpg")
    obj.get_blue_red()
    obj.set_dict("top_vardi.json")
    obj.show_bar_plot()
