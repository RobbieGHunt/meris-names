# -*- coding: utf-8 -*-
"""
A short machine learning project.

Takes in a training set of text, trains the model, then creates funny names.

Follows the example of https://www.tensorflow.org/text/tutorials/text_generation,
adapted for nefarious purposes.
https://jovian.ai/joshuatan777/tensorflow-tutorial-text-generation-character-based
seems to work for some hangups in the official version.

May need to change how names are sampled - in this code they arbitrarily define it
to be 64 characters. We could define it by splitting based on \n options.

Flowchart:
    - read in names as a text script
    - vectorize names into ids
    - create a dataset of the ids split by \n characters (individual names). Each name will
        act as a sequence rather than a random amount of characters. Amount of names will be
        batches.
    - Build a model. Base this off of the one in the tutorial.
    - Check losses, a NaN loss means the model will not work.
    - Training, figure out checkpoints and what good settings will be. 

@author: R. G. Hunt
"""

import tensorflow as tf
import numpy as np
import os
import time


from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

#Tk gui to choose the training data.

root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
root.wm_attributes('-topmost', 1)
filename = askopenfilename(parent=root) # show an "Open" dialog box and return the path to the selected file

print(filename)

#Reads in the text.
text = open(filename, 'rb').read().decode(encoding='utf-8')

#A list of all the unique characters.
vocab = sorted(set(text))

example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

#string lookup layer that converts tokens to IDs.
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

#And the reverse functions that converts IDs to characters.
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary = ids_from_chars.get_vocabulary(), invert=True,
    mask_token=None)


def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

#turns ids back into individual characters, prints them up to take(number).
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))







