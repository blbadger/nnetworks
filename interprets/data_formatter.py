# data_formatter.py

# import standard libraries
import random

# import third-party libraries
import torch
import pandas as pd 
import numpy as np 


class Format:

	def __init__(self, file, prediction_feature, training=True):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower()[0] == 'n' else x)
		df = df[:][:10000]
		length = len(df[:])
		field_of_interest = prediction_feature
		self.prediction_feature = prediction_feature

		if training:
			# 80/20 training/test split and formatting
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[[i for i in training.columns if i != prediction_feature]]
			self.training_outputs = training[[prediction_feature]]

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.val_inputs = validation[[i for i in validation.columns if i != prediction_feature]]
			self.val_outputs = validation[[prediction_feature]]

			self.sequential_count = 0
			self.val_inputs.reset_index(inplace=True)
			self.val_outputs.reset_index(inplace=True)

		else:
			self.test_inputs = df 



	def stringify_input(self, index, input_type='training', short=False, n_taken=4, remove_spaces=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input
		kwargs:
			input_type: str, type of data input requested ('training' or 'test' or 'validation')
			short: bool, if True then at most n_taken letters per feature is encoded
			n_taken: int, number of letters taken per feature
			remove_spaces: bool, if True then input feature spaces are removed before encoding

		Returns:
			array: string: str of values in the row of interest

		"""

		if input_type == 'training':
			inputs = self.training_inputs

		elif input_type == 'validation':
			inputs = self.val_inputs

		else:
			inputs = self.test_inputs

		if short == True:
			inputs = inputs.applymap(lambda x: '_'*n_taken if str(x) in ['', '(null)'] else str(x)[:n])
		else:
			inputs = inputs.applymap(lambda x:'_'*n_taken if str(x) in ['', '(null)'] else str(x))

		inputs = inputs.applymap(lambda x: '_'*(n_taken-len(x)) + x)

		i = index
		string_arr = inputs.apply(lambda x: '_'.join(x.astype(str)), axis=1)

		return string_arr


	@classmethod
	def string_to_tensor(self, string):
		"""
		Convert a string into a tensor

		Args:
			string: str

		Returns:
			tensor: torch.Tensor

		"""

		places_dict = {s:i for i, s in enumerate('0123456789.:- ')}

		# vocab_size x embedding dimension (ie input length)
		tensor_shape = (len(string), len(places_dict)) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(string):
			print (letter)
			tensor[i][places_dict[letter]] = 1.

		tensor = torch.flatten(tensor)
		return tensor


	def transform_to_tensors(self):
		"""
		Transform input and outputs to arrays of tensors

		"""

		input_arr, output_arr = [], []
		string_arr = []
		for i in range(len(self.training_outputs[self.prediction_feature])):
			if self.training_outputs[self.prediction_feature][i]:
				input_string = self.stringify_input(i)
				input_arr.append(self.string_to_tensor(input_string))
				output_arr.append(torch.tensor(self.training_outputs[self.prediction_feature][i]))

		return input_arr, output_arr


	def validation(self):
		"""
		Choose one example from the test set, such that repeated
		choices iterate over the entire set.

		Args:
			None

		Returns:
			output: float
			input str
			output_tensor: torch.Tensor()
			input_tensor: torch.Tensor()

		"""

		index = self.sequential_count
		output = self.val_outputs['etime'][index]
		output_tensor = torch.tensor(output)

		input_string = self.stringify_input(index, training=False)
		input_tensor = self.string_to_tensor(input_string)
		self.sequential_count += 1

		return output, input, output_tensor, input_tensor


	def generate_test_inputs(self):
		"""
		Generate tensor inputs from a test dataset

		"""
		inputs = []
		for i in range(len(self.test_inputs)):
			input_tensor =self.string_to_tensor(input_string)
			inputs.append(input_tensor)

		return inputs



