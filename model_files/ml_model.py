import pandas as pd
import numpy as np 
import string
import tensorflow as tf
import random
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback


dwarves = open("model_files/Dwarf Names Training Data.txt","r")
dwarf_names = []
for dwarf in dwarves:
    dwarf_stripped = dwarf.rstrip()
    dwarf_stripped = dwarf_stripped.lower()
    dwarf_names.append(dwarf_stripped)

dwarf_names = list(map(lambda s: s + '.', dwarf_names))

dwarves.close()

chars = string.ascii_lowercase
chars_num = len(chars)

max_length = len(max(dwarf_names, key=len))
dwarf_length = len(dwarf_names)

max_length = len(max(dwarf_names, key=len))
dwarf_length = len(dwarf_names)
char_to_id = {c:i for i, c in enumerate(chars)}
char_to_id['.'] = 26

id_to_char = {v:k for v,k in enumerate(chars)}
id_to_char[26] = '.'

char_dim = len(char_to_id)


##functions
def make_name(model,user_string):
    name = []
    x = np.zeros((1, max_length, char_dim))
    end = False
    i = 0
    any_length = random.randint(5,10)
    name.append(user_string)
    
    while end==False:
        probs = list(model.predict(x)[0,i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(char_dim), p=probs)
        if i == max_length-2:
            character = '.'
            end = True
        else:
            character = id_to_char[index]
        name.append(character)
        x[0, i+1, index] = 1
        i += 1
        if character == '.':
            end = True
    
    return ''.join(name)