#include <iostream>
#include <iomanip>
#include <random>
#include "matrix_operations.cpp"

using namespace std;

vector<float> architecture = {3, 2};
vector<vector<float>> weights = {{3, 2, 4}, {2, 1, 4}};
vector<vector<float>> biases = {{2, 1}};

vector<float> activation_function(vector<float> z_array){
		vector<float> sigmoid_arr = sigmoid(z_array); // from matrix_operations.cpp
		return sigmoid_arr;
	};

vector<float> network_output(vector<float> output){
	for (int i=0; i < architecture.size(); i++){
		vector<vector<float>> weight_arr {weights[i]};
		vector<vector<float>> biases_arr {biases[i]};
		cout << weight_arr.size();
		//for (int i=0; i < weight_arr.size(); i++){
			//for (int j=0; j < weight_arr[i].size(); j++){
				//cout << weight_arr[i][j];
			//}
		//}
		//vector<vector<float>> z_vec = matmult(transpose(weight_arr), output);
		//vector<float> activations = mat_add(z_vec, biases_arr);
		//output = activation_function(activations);
		//}
	}
	return output;
	
}
	
int main(){
	vector<float> output = {{1, 2, 3}};
	vector<float> fin;
    fin = network_output(output);
	for (auto u: fin) cout << u << " ";
}
