import torch
import time
import html
import webbrowser
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (f'Device: {device}')

class GPTEval:

	def __init__(self, model, input_sequence):
		self.input_sequence = input_sequence
		input_ids = tokenizer.encode(
			input_sequence,
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			padding=False
			)

		self.input_ids = input_ids.to(device)
		self.model = model


	def gradientxinput(self, normalized=True):
		"""
		Computes input saliency using gradient wrt. input
		"""

		indicies_arr = []
		input_tensor = self.input_ids
		position_ids = torch.tensor([i for i in range(len(self.input_ids))])

		# assemble input embeddings as leaf variables
		with torch.no_grad():
			assert torch.equal(self.model.transformer.wte(input_tensor), self.model.get_input_embeddings()(input_tensor))
			embedded_inputs = torch.clone(self.model.transformer.wte(input_tensor))


		embedded_inputs.requires_grad = True
		outputs = self.model(input_embeds=embedded_inputs)
		total_output = torch.sum(outputs[0][:, -1, :])

		# backpropegate output gradient to the input embedding
		total_output.backward()
		saliency_arr = torch.abs(embedded_inputs.grad * embedded_inputs)
		saliency_arr = torch.sum(saliency_arr[0], dim=1)

		# absmax norm
		if normalized:
			if torch.max(saliency_arr) != 0:
				correction_factor = 1 / torch.max(saliency_arr)
				saliency_arr *= correction_factor

		indicies_arr = [i for i in range(len(saliency_arry))]
		return indicies_arr, saliency_arr


	@torch.no_grad()
	def occlusion(self, normalized=True):
		"""
		Generates a perturbation-sytpe attribution using occlusion.  Not suitable for long input sequences
		"""

		# assumes no batching for self.input_ids
		output = self.model(self.input_ids)
		input_tensor = self.input_ids
		occlusion_arr, indicies_arr = [], []

		for i, ele in enumerate(self.input_ids[0]):
			input_copy = torch.clone(input_tensor)
			input_copy[0, i] = tokenizer.encode(tokenizer.eos_token)[0] # zero out the token of interest

			output_missing = self.model(input_copy)
			# l1 distance between output and modified output for the last word
			difference = float(torch.sum(torch.abs(output_missing[0][:, -1, :] - output[0][:, -1, :])))
			occlusion_arr.append(difference)
			indicies_arr.append(i)

		assert len(occlusion_arr) == len(self.input_ids[0])
		if normalized:
			if max(occlusion_arr) != 0:
				correction_factor = 1/max(occlusion_arr)
				occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return indicies_arr, occlusion_arr


	def readable_interpretation(self, decoded_input, metric='combined'):
		"""
		Determines relative importance for each input element
		"""

		importances_arr = []

		if metric == 'combined':
			x, y1 = self.occlusion(normalized=True)
			x, y2 = self.occlusion(normalized=True)
			y = [(a + b)/2 for a, b in zip(y1, y2)]


		elif metric == 'occlusion':
			x, y = self.occlusion(normalized=True)

		elif metric == 'gradientxinput':
			x, y = self.gradientxinput(normalized=True)

		else:
			raise ValueError
	
		summed_ems = torch.tensor(y)

		# zeropoint  
		if torch.max(summed_ems) > torch.min(summed_ems):
			summed_ems = (summed_ems - torch.min(summed_ems)) / (torch.max(summed_ems) - torch.min(summed_ems))

		positions = [i for i in range(len(summed_ems))]

		# assemble HTML file with red (high) to blue (low) attributions per token
		highlighted_text = []
		for i in range(len(positions)):
			word = decoded_input[i]
			red, green, blue = int((summed_ems[i]*255)), 110, 110
			color = '#{:02x}{:02x}{:02x}'.format(red, green, blue)
			highlighted_text.append(f'<span style="background-color: {color}">{word}</span>')

		with torch.no_grad():
			embedded_inputs = torch.clone(self.model.transformer.wte(self.input_ids))
			output = self.model(inputs_embeds=embedded_inputs)[0][:, -1, :]
			predicted_word = self.model(self.input_ids)[0][:, -1, :] # should be equal to output
			assert torch.equal(output, predicted_word)

			predicted_word = int(torch.argmax(predicted_word, dim=1))
			predicted_word = tokenizer.decode(predicted_word)

		highlighted_text = ' '.join(highlighted_text)
		highlighted_text += f'</br> </br> Predicted next word: {predicted_word}'
		with open('data.html', 'wt', encoding='utf-8') as file:
			file.write(highlighted_text)
		webbrowser.open('data.html')

		return summed_ems

load_8bit = False
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print ('tokenizer downloaded or loaded from cache')

model = AutoModelForCausalLM.from_pretrained("gpt2", load_in_8bit=load_8bit, device_map='auto')
print ('model downloaded or loaded from cache') # or "gpt2-xl" etc.

model.eval()

if __name__ == '__main__':
	input_sequence = 'The wipers on the bus go swish swish'
	input_sequence = input_sequence.strip()
	gevaluate = GPTEval(model, input_sequence)
	input_ids = gevaluate.input_ids[0]
	decoded_input = []
	for symbol in input_ids:
		decoded_input.append(tokenizer.decode(symbol))

	arr = gevaluate.readable_interpretation(decoded_input, metric='combined')
