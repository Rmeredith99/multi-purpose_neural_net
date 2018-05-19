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

def get_chord_info(chord):
	"""
	[get_chord_info] is data regarding note, octave, and
		duration for all individual notes in the chord.
		Returns: string representing a chord object
	[chord] - a music21 chord object
	"""
	pitches = chord.pitches
	duration = chord.duration.quarterLength
	s = "Chord: \n"
	for n in chord:
		s += ("\t %s%d %0.1f" % (n.pitch.name, n.pitch.octave, duration)) + "\n"
	return s

def duration(sound):
	"""
	[duration] is a float that represents how long a
		note is held for.
		Ex: quarter note -> 0.25
			dotted half note -> 0.75
		Returns: int representing sound duration
	[sound] : a music21 note, chord, or rest object
	"""
	note_type = sound.duration.type
	dots = sound.duration.dots
	dot_scale_factor = 1.0
	
	#adjusting for dotted notes
	for i in range(dots):
		dot_scale_factor += 0.5**i
		
	if note_type == "64th":
		duration = 1.0/64
	elif note_type == "32th":
		duration = 1.0/32
	elif note_type == "16th":
		duration = 1.0/16
	elif note_type == "eighth":
		duration = 1.0/8
	elif note_type == "quarter":
		duration = 1.0/4
	elif note_type == "half":
		duration = 1.0/2
	elif note_type == "whole":
		duration = 1.0/1
	else:
		duration = 2.0
		
	return tanh(duration*dot_scale_factor)

def note_vector(note):
	"""
	[note_vector] is a numpy array with 89 entries that
		denotes a one-hot representation of the note.
		The 89th entry is for 'rest'.
		Returns: length 89 numpy array
	[note] : a music21 note object 
	"""
	vector = np.array([0]*89,dtype=float)
	octave = int(note.pitch.octave)
	name = (note.pitch.name).replace("-","")
	is_sharp = (name[-1]=="#")
	possible_notes = {"C":0,"C#":1,"D":2,"D#":3,"E":4,"F":5,"F#":6,"G":7,"G#":8,"A":9,"A#":10,"B":11}
	note_index = -9 + octave*12 + possible_notes[name]
	vector[note_index] = 1#duration(note)
	return vector

def rest_vector(rest):
	"""
	[rest_vector] is a numpy array with 89 entries that
		denotes a one-hot representation of the note.
		The 89th entry is for 'rest'.
		Returns: length 89 numpy array
	[rest] : a music21 rest object 
	"""
	vector = np.array([0]*89,dtype=float)
	vector[88] = 1#duration(rest)
	return vector

def chord_vector(chord):
	"""
	[chord_vector] is a numpy array with 89 entries that
		denotes a one-hot representation of all the notes
		in the chord. The 89th entry is for 'rest'.
		Returns: length 89 numpy array
	[chord] : a music21 chord object 
	"""
	vector = np.array([0]*89,dtype=float)
	for note in chord:
		vector = vector + note_vector(note)
	return vector
	
def vectorize(song_file):
	"""
	[vectorize] is a 2D matrix whose rows represent note,
		rest, and chord vectors. The dimesions are n x 89
		where n is the number of distinct sounds in the
		song.
		Returns: (unique sounds x 89) numpy array
	[song_file] : string - a midi file in the working directory	
	"""
	
	song = converter.parse(r'Midi_files2\%s' %(song_file))
	
	vectors = []
	
	for n in song.flat:
		try:
			if n.isNote:
				#print ("Note: %s%d %0.1f" % (n.pitch.name, n.pitch.octave, n.duration.quarterLength))
				#print ("Note: %s%d type:%s dots:%s" % (n.pitch.name, n.pitch.octave, n.duration.type,n.duration.dots))
				#print ("")
				vectors.append(note_vector(n))
			elif n.isRest:
				#print ("Rest: %0.1f" % (n.duration.quarterLength))
				#print ("")
				if n.duration.quarterLength<5:
					vectors.append(rest_vector(n))
			elif n.isChord:
				#print (get_chord_info(n))
				vectors.append(chord_vector(n))
		except AttributeError as e:
			pass

	return (np.array(vectors))

def write_to_file(song_file):
	"""
	[write_to_file] writes the vector matrix for the
		given midi file to an external text document
		with the same name as the input file.
		Returns: None
	[song_file] : string - a midi file in the working directory
	"""
	file_name = "Midi_output\\"+song_file[:-3] + "txt"
	
	vector_matrix = vectorize(song_file)
	n,_ = vector_matrix.shape
	with open(file_name,"w") as F:
		for i,vector in enumerate(vector_matrix):
			if i == n-1:
				F.write(str(vector))
			else:
				F.write(str(vector) + "\n")
				
		
def get_data(directory,n,samples=1):
	"""
	[get_data] returns an input/output tuple that will
		will be used for training purposes. There will
		be [n] concatenated vectors for the input vector
		which correspond to the previous [n] notes. The
		output vector will be the following note. The
		samples will be taken from vector matrix.
		Returns: (samples x 89*n) numpy array,
					(samples x 89) numpy array
	[directory] : string - directory of the midi files from which
		to sample
	[n] : int - the number of preceding notes used to determine
		the next note
	[samples] : int - the number of input/outputs returned (set
		to a default value of 1) (OPTIONAL)
	"""
	vector_matrix_list = []
	i = 1
	for song in os.listdir(directory):
		print (str(i) + " of 96")
		if i==20:
			break
		vector_matrix_list.append((vectorize(song)))
		i += 1
	
	#array for collecting samples
	input_array = [] # (samples x (89*n))
	output_array = [] # (samples x 89)
	song_index_distribution = []
	
	for i,e in enumerate(vector_matrix_list):
	    song_index_distribution.extend([i]*(len(e)))
	
	for i in range(samples):
		#randomly selecting one song
		num_songs = len(vector_matrix_list)
		song_index = song_index_distribution[randint(0,len(song_index_distribution)-1)]
		vector_matrix = vector_matrix_list[song_index]
		
		#randomly selecting one part of that song
		matrix_length = vector_matrix.shape[0]
		output_index = randint(n,matrix_length-1)
		input_data = np.ravel(vector_matrix[output_index-n:output_index])
		output_data = vector_matrix[output_index]
	
		input_array.append(input_data)
		output_array.append(output_data)
		
	return np.array(input_array), np.array(output_array)
	
def set_up_NN(n, weights=None):
	"""
	[set_up_NN] creates the neural network and changes the global
		variable 'model' to the network that is established in
		this function.
		Returns: None
	[n] : int - the number of previous notes that we will use as input
	[weights] : string - a .h5 file containing the saved weights for this
		neural network (OPTIONAL)
	"""
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	
	input_nodes = 89*n
	
	def rmse(y_true, y_pred):
		return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
	
	global model
	model = Sequential()
	model.add(Dense(input_nodes,input_dim=input_nodes,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(500,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(500,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(200,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(100,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(89,activation="softmax"))
	
	"""[loss] is set to mean_squared_error becuase then it can have values that aren't
	integers when training. Similarly, [metrics] is set to 'rmse' which means
	root_mean_squared_error, and this is for evaluating the results.
	"""
	#model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

	
	if weights != None:
		model.load_weights(weights)
	
def train(n,samples,weights,epochs=200,batch_size=50):
	"""
	[train] takes in a desired number of training samples
		and trains the globally-defined neural network
		using that data
		Returns: None
	[n] : int - the number of previous notes that we will use as input
	[samples] : int - an integer number of training samples
	[weights] : string - a .h5 file name to save the trained
		weights to
	[epochs] : int - number of times the NN runs over the same chunk
		of data (OPTIONAL)
	[batch_size] : int - number of I/O data trained on before updating
		(OPTIONAL)
	"""
	print("Begin Collecting Data")
	x,y = get_data(data_dir,n,samples)
	print("Done Collecting Data")
	
	global model
	model.fit(x,y,epochs=epochs,batch_size=batch_size)
	model.save_weights(weights)
	
def predict(input_vector):
	"""
	[predict] is the output of running [input_vector] through the
		globally defined nerual network.
		Returns: 89 length numpy array
	[input_vector] : numpy array - an array of vectors representing
		the last n notes played
	"""
	return model.predict(input_vector)
	
def clean_vector(vector):
	"""
	[clean_vector] takes in an output from the neural network and
		makes it look more like a single note, chord, or rest instead
		of noise.
		Returns: length 89 numpy array
	[vector] : numpy array - representation of a musical sound
	"""
	output_vector = np.zeros(89,dtype=float)
	std = np.std(vector)
	mean = np.mean(vector)
	
	new_vector = np.clip(vector,0,1)
	argmax = np.argmax(new_vector)
	max_val = np.max(new_vector)
	
	# if rest is max then it's the only thing
	if argmax == 88:
		output_vector[88] = .5
		return output_vector * vector
	
	for i in range(89):
		if vector[i]+std > max_val:
			output_vector[i] = .5
		
	# a rest note can't be part of a chord
	output_vector[88] = 0.
	output_vector = output_vector #* new_vector
	
	return output_vector
	
	
def generate_song(n,length,buff=30):
	"""
	[generate_song] is a numpy array containing a vector representation
		of sounds (notes, chords, and rests).
		Returns: (length x 89) numpy array
	[n] : int - the number of previous notes that we will use as input
	[length] : int - the number of sounds in the song
	[buff] : int - length of buffer at the beginning of the song to
		counter the effects of the random start (OPTIONAL)
	"""
	def make_random_vector():
		v = np.zeros(89,dtype=float)
		notes = randint(1,3)
		duration = uniform(.0,1.0)
		for i in range(notes):
			v[randint(0,87)] = duration
		return v
		#return get_data(data_dir,10,samples=1)[0]
		pass
	
	song = []
	song = list(get_data(data_dir,n,samples=1)[0].reshape(n,89))

	for i in range(n):
		#song.append(make_random_vector())
		pass

	for i in range(length + buff):
		temp_song = song[i+1:i+n-1]
		temp_song.append(make_random_vector())
		temp_song.insert(0,make_random_vector())
		#input_vector = np.ravel(np.array(song[i:i+n-1]))
		input_vector = np.ravel(np.array(temp_song))
		new_vector = predict(np.array([input_vector]))[0]
		song.append(clean_vector(new_vector))
	
	return np.array(song[buff+n:])

def vector_to_sound(vector):
	"""
	[vector_to_sound] is a music21 object (note, chord, or rest) that
		is derived from a given vector.
		Returns: music21.note, music21.chord, or music21.rest
	[vector] : numpy array - a length 89 numpy array that represents
		a musical sound
	"""
	possible_notes = {0:"C",1:"C#",2:"D",3:"D#",4:"E",5:"F",6:"F#",7:"G",8:"G#",9:"A",10:"A#",11:"B"}
	indices = []
	value = 0.
	for i in range(89):
		if vector[i] > 0:
			indices.append(i)
			value = (vector[i]) * 1.5 #extra factor is just to sound better

	def index_to_note(i):
		if i == 88:
			return "R"
		i += 9
		octave = str(int(i/12))
		n = possible_notes[i%12]
		return (n + octave)
	
	if 88 in indices:
		return note.Rest(quarterLength = value)
	
	if len(indices)>1:
		note_list = []
		for i in indices:
			note_list.append(index_to_note(i))
		return chord.Chord(note_list,quarterLength = value)
	
	else:
		i = indices[0]
		n = index_to_note(i)
		return note.Note(n, quarterLength = value)
	
def to_midi(song,filename):
	"""
	[to_midi] writes a midi file of the given song array to the
		current directory with name [filename].
		Returns: None
	[song] : numpy array - (length x 89) numpy array representing
		a given song using the vector notation
	[filename] : string - name that the file will be saved under
	"""
	s = stream.Stream()
	
	for vector in song:
		s.append(vector_to_sound(vector))
				 
	mf = midi.translate.streamToMidiFile(s)
	mf.open(filename+".mid",'wb')
	mf.write()
	mf.close()

if __name__ == "__main__":
	"""
	music.h5  : original dataset (not useful)
	music2.h5 : original dataset (some good music then repeats)
	music3.h5 : expanded dataset (never repeats, but random music)
	music4.h5 : expanded dataset (some good music, doesn't repeat)
	music5.h5 : expanded dataset, last 30 notes (not good, but not terrible)
	"""
	global model
	global data_dir
	data_dir = "Midi_files2"
	
	last_n_notes = 20
	
	# establishing the NN
	set_up_NN(last_n_notes, weights="music6.h5")
	song = generate_song(last_n_notes, 100, buff=0)
	#print(song)
	to_midi(song,"midi_example_NN6_10")
	#train(last_n_notes,100000,"music6.h5",epochs=25,batch_size=64)
	
	
