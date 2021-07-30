#include <iostream>
#include <iomanip>
#include <random>
#include "matrix_operations.cpp"

using namespace std;

vector<float> architecture = {3, 2, 2};
vector<vector<vector<float>>> weights = {{{3, 2, 4}, 
										  {2, 1, 0}},
										 {{-1, 2},
										  {0.5, 1}}};
										  
vector<vector<vector<float>>> biases = {{{2},
										 {0}},
										{{1},
										 {-1}}};

vector<vector<float>> activation_function(vector<vector<float>> z_array){
		vector<vector<float>> sigmoid_arr = sigmoid(z_array); 
		return sigmoid_arr;
	};
	
vector<vector<vector<float>>> z_vectors;

//weight, bias = self.weights[index], self.biases[index]
			//output = self.activation_function(np.dot(weight.T, output) + bias)
			

vector<vector<float>> network_output(vector<vector<float>> output){
	//weights_init();
	//print_weights();
	
	
	for (int i=0; i < architecture.size() - 1; i++){
		vector<vector<float>> weight_arr = weights[i];
		vector<vector<float>> biases_arr = biases[i];
		vector<vector<float>> transposed_weight = transpose(weight_arr);
		
		//for (int i=0; i < transposed_weight.size(); i++){
			//for (int j=0; j < transposed_weight[i].size(); j++){
				//cout << transposed_weight[i][j] << " ";
			//}
			//cout << "\n";
		//}

		vector<vector<float>> z_vec = matmult(weight_arr, output);
		
		vector<vector<float>> activations = matadd(z_vec, biases_arr);
		output = activation_function(activations);
		
	}
	return output;
}

vector<vector<vector<float>>> backpropegate(vector<vector<float>> output, vector<vector<float>> classification){
	vector<vector<vector<float>>> activations_arr;
	
	for (int i=0; i < architecture.size() - 1; i++){
		vector<vector<float>> weight_arr = weights[i];
		vector<vector<float>> biases_arr = biases[i];
		vector<vector<float>> transposed_weight= transpose(weight_arr);

		vector<vector<float>> z_vec = matmult(weight_arr, output);
		vector<vector<float>> activations = matadd(z_vec, biases_arr);
		
		output = activation_function(activations);
		activations_arr.push_back(output);
		z_vectors.push_back(activations);
	}
	
	return activations_arr;
}


vector<vector<float>> activation_prime(vector<vector<float>> z){
	vector<vector<float>> zac = activation_function(z);
	vector<vector<float>> negz = scalar_mult(z, -1);
	vector<vector<float>> res = hadamard(zac, scalar_add(negz, 1));
	return res;
};


vector<vector<float>> cost_function_derivative(vector<vector<float>> output_activations, vector<vector<float>> y){
	vector<vector<float>> neg_y = scalar_mult(y, -1);
	vector<vector<float>> res = matadd(output_activations, neg_y);
	return res;
};
	
int main(){
	vector<vector<float>> output = {{1},
									{0},
									{-1}};
								
	vector<vector<float>> classification = {{0.}, {1.}};						
	vector<vector<vector<float>>> activations_arr = backpropegate(output, classification);
	vector<vector<float>> last_acts = activations_arr[activations_arr.size()-1];
	
	vector<vector<float>> last_prime = activation_prime(last_acts);
	vector<vector<float>> error = hadamard(cost_function_derivative(last_acts, classification), last_prime);
	
	vector<vector<float>> transposed_weight = error;
	for (int i=0; i < transposed_weight.size(); i++){
			for (int j=0; j < transposed_weight[i].size(); j++){
				cout << transposed_weight[i][j] << " ";
			}
			cout << "\n";
		}
	
		 
	vector<vector<vector<float>>> fin = z_vectors;
	

	for (int i=0; i < fin.size(); i++){
		for (int j=0; j < fin[i].size(); j++){
			for (int k=0; k < fin[i][j].size(); k++){
				cout << fin[i][j][k];
			}
			cout << " ";
		}
		cout << "\n";
	}
	return 0;
}
