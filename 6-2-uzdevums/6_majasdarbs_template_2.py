#!/usr/bin/env python3
'''
Python 6 nodarbības mājasdarbs Nr.2

Uzdevums: aizpildīt vietas ar atzīmi TODO
'''
import matplotlib.pyplot as plt
import json
import numpy as np

# Importēt failu "top_vardi.json" un saglabāt atslēgas kā listi ar nosaukumu "x"
# vērtības kā listi ar nosaukumu "y"

top_vardi_dict: dict = json.loads(open('top_vardi.json').read())

x = top_vardi_dict.keys()
y = top_vardi_dict.values()

# izveidot stabiņveidu grafiku kas rāda vārdu biežumu (y ass), Vārdus uz x ass
# piemērs ir mājasdarbu failā
fig, ax = plt.subplots()
ax.bar(x,y)

plt.show()