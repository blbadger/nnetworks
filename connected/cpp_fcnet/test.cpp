#include <iostream>
#include <iomanip>
#include <random>
#include "matrix_operations.cpp"

using namespace std;

vector<float> architecture = {3, 2};
vector<vector<vector<float>>> weights = {{{3, 2, 4}, 
										  {2, 1, 0}}};
										  
vector<vector<vector<float>>> biases = {{{2},
										 {0}}};

vector<vector<float>> activation_function(vector<vector<float>> z_array){
		vector<vector<float>> sigmoid_arr = sigmoid(z_array); 
		return sigmoid_arr;
	};

//weight, bias = self.weights[index], self.biases[index]
			//output = self.activation_function(np.dot(weight.T, output) + bias)

vector<vector<float>> network_output(vector<vector<float>> output){
	for (int i=0; i < architecture.size() - 1; i++){
		vector<vector<float>> weight_arr = weights[i];
		vector<vector<float>> biases_arr = biases[i];
		vector<vector<float>> transposed_weight = transpose(weight_arr);
		
		for (int i=0; i < transposed_weight.size(); i++){
			for (int j=0; j < transposed_weight[i].size(); j++){
				cout << transposed_weight[i][j] << " ";
			}
			cout << "\n";
		}

		vector<vector<float>> z_vec = matmult(weight_arr, output);
		for (int i=0; i < z_vec.size(); i++){
			for (int j=0; j < z_vec[i].size(); j++){
				cout << z_vec[i][j] << " ";
			}
			cout << "\n";
		}
		
		vector<vector<float>> activations = matadd(z_vec, biases_arr);
		//for (int i=0; i < activations.size(); i++){
			//for (int j=0; j < activations[i].size(); j++){
				//cout << activations[i][j] << " ";
			//}
			//cout << "\n";
		//}
		output = activation_function(activations);
		
	}
	return output;
}
	
int main(){
	vector<vector<float>> output = {{1},
									{0},
									{-1}};
									
	vector<vector<float>> fin = network_output(output);

	for (int i=0; i < fin.size(); i++){
		for (int j=0; j < fin[i].size(); j++){
			cout << fin[i][j];
			cout << "\n";
		}
	}
	return 0;
}
