#include <iostream>
#include <iomanip>
#include <random>
#include "matrix_operations.cpp"

using namespace std;

vector<float> architecture = {3, 2};
vector<vector<vector<float>>> weights = {{{3, 2, 4}, 
										  {2, 1, 4}}};
vector<vector<vector<float>>> biases = {{{2, -1}}};

vector<vector<float>> activation_function(vector<vector<float>> z_array){
		vector<vector<float>> sigmoid_arr = sigmoid(z_array); 
		return sigmoid_arr;
	};

vector<vector<float>> network_output(vector<vector<float>> output){
	for (int i=0; i < architecture.size() - 1; i++){
		vector<vector<float>> weight_arr = weights[i];
		vector<vector<float>> biases_arr = biases[i];
		vector<vector<float>> transposed_output = transpose(output);
		//for (int i=0; i < transposed_output.size(); i++){
			//for (int j=0; j < transposed_output[i].size(); j++){
				//cout << transposed_output[i][j] << " ";
			//}
			//cout << "\n";
		//}

		vector<vector<float>> z_vec = matmult(weight_arr, transposed_output);
		vector<vector<float>> activations = matadd(transpose(z_vec), biases_arr);
		output = activation_function(activations);
		
	}
	return output;
}
	
int main(){
	vector<vector<float>> output = {{1, 2, 3}};
	vector<vector<float>> fin;
    fin = network_output(output);
	for (int i=0; i < fin.size(); i++){
			for (int j=0; j < fin[i].size(); j++){
				cout << fin[i][j] << " ";
			}
			cout << "\n";
		}
}
