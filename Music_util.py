from music21 import *
from math import tanh
from scipy import arctanh
import numpy as np
from random import randint, uniform

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras import backend

import os

import Music

def from_music_input(x):
	"""
	[from_music_input] converts an input array for music
		generation to an input array for the multi-purpose
		neural network.
		Returns: numpy array that can be put into the MPNN
	[x] : numpy array - an array that will contain a number
		of inputs for the music generator.
		Shape: (n x 1780)
	"""
	n,_ = x.shape
	output = np.zeros((n,816+1780), dtype = float)
	for i in range(n):
		output[i,816:] = x[i]
	return output

def from_music_output(y):
	"""
	[from_music_output] converts an output array for music
		generation to an output array for the multi-purpose
		neural network.
		Returns: numpy array that is the result of the MPNN
	[y] : numpy array - an array that will contain a number
		of outputs for the music generator.
		Shape: (n x 89)
	"""
	n,_ = y.shape
	output = np.zeros((n,26+89), dtype = float)
	for i in range(n):
		output[i,26:] = y[i]
	return output

def to_music_input(x):
	"""
	[to_music_input] converts from an input array for the
		multi-purpose neural network to an input array for
		music generation.
		Returns: numpy array that can be put into the music
			generator
	[x] : numpy array - an array that will contain a number
		of inputs for the MPNN.
		Shape: (n x 2596)
	"""
	n,_ = x.shape
	output = np.zeros((n,1780), dtype = float)
	for i in range(n):
		output[i,:] = x[i,816:]
	return output

def to_music_output(y):
	"""
	[to_music_output] converts from an output array for the
		multi-purpose neural network to an output array for
		music generation.
		Returns: numpy array that can be put into the music
			generator
	[y] : numpy array - an array that will contain a number
		of inputs for the MPNN.
		Shape: (n x 115)
	"""
	n,_ = y.shape
	output = np.zeros((n,89), dtype = float)
	for i in range(n):
		output[i,:] = y[i,26:]
	return output

def get_music_data(n):
	"""
	[get_music_data] is a tuple of input/output data meant to
		be used with the multi-purpose neural network.
		Returns: tuple of numpy arrays (input,output)
	[n] : int - number of samples
	"""
	x_data, y_data = Music.get_data("Midi_files2",20,n)
	return from_music_input(x_data), from_music_output(y_data)