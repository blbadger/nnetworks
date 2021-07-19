#include <iostream>
#include <iomanip>
#include <random>
#include "matrix_operations.cpp"

using namespace std;

vector<float> architecture = {3, 2};
vector<vector<vector<float>>> weights = {{{3, 2, 4}, 
										  {2, 1, 0}}};
										  
vector<vector<vector<float>>> biases = {{{2, 0}}};

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
		for (int i=0; i < activations.size(); i++){
			for (int j=0; j < activations[i].size(); j++){
				cout << activations[i][j] << " ";
			}
			cout << "\n";
		}
		output = activation_function(activations);
		
	}
	return output;
}
	
int main(){
	vector<vector<float>> output = {{1, 0.4, -1}};
	weights.push_back(output);
	vector<vector<vector<float>>> fin = weights;
	//cout << fin[1][0][2];
	cout << fin[0][0].size();
	
    //fin = network_output(output);
    //fin = scalar_mult(output, 2);
	//for (int i=0; i < fin.size(); i++){
		//for (int j=0; j < fin[i].size(); j++){
			//for (int k=0; i < fin[i][j].size(); k++) {
				//cout << i << j << k << " ";
			//}
		//}
			//cout << "\n";
	//}
	return 0;
}
