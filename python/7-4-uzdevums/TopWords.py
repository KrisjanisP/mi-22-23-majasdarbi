#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json

'''
Python 7 mājasdarbs Nr.2

Uzdevums: aizpildīt vietas ar atzīmi TODO

Izveidot klasi, kura pārveido 5. nodarbības mājasdarbu Nr. 2 saturu par klasi

'''

class TopWords:
    '''
    Izveidot klasi, kurai ir 2 publiskas metodes:
    - setVardnica -  definē failu
    - grafiks - izvada grafiku

    Klasei nav pieejami publiski parametri
    '''
    def set_dict(self,filename):
        self.word_dict = dict()
        with open(filename,'r',encoding='utf-8') as file:
            self.word_dict = json.load(file)

    def show_bar_plot(self):
        plt.bar(range(len(self.word_dict)), list(self.word_dict.values()), align='center')
        plt.xticks(range(len(self.word_dict)), list(self.word_dict.keys()))
        plt.show()


if __name__ == "__main__":
    obj = TopWords()
    obj.set_dict("top_vardi.json")
    obj.show_bar_plot()
