
// fc_network.cpp
/* 
 * A neural network for MNIST digit classification. Sigmoid activation
 * function, stochastic gradient descent
 * 
 * */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

using namespace std;

class Network{
	vector<int> architecture = {784, 20, 10};
	int network_length = architecture.size();
	
	// initialize biases
	vector<vector<float>> biases_init(){
		vector<vector<float>> biases;
		for (int i=0; i < network_length-1; i++){
			vector<float> temp;
			for (int j=0; j < architecture[i]; j++){
				//normal_distribution<> val{0,1};
				float val = 0;
				temp.push_back(val);
			}
			biases.push_back(temp);
		}
		return biases;
	}
	
	// initialize weights
	vector<vector<vector<float>>> weights_init(){
		vector<vector<vector<float>>> weights;
		for (int i=0; i < network_length-1; i++){
			vector<vector<float>> layer;
			for (int j=0; j < architecture[i]; j++){
				vector<float> temp;
				for (int k=0; k < architecture[i]; k++){
					float val=0;
					temp.push_back(val);
				}
				layer.push_back(temp);
			}
			weights.push_back(layer);
		}
		return weights;
	}
	vector<vector<float>> biases = biases_init();
	vector<vector<vector<float>>> weights = weights_init();
	
	public:
	void print_weights(){
		for (auto u: weights){
			for (auto v: u){
				for (auto q:v){
					cout << q;
				}
			}
		}
	}
	
	float activation_function(float z){
		float sigmoid_z = 1 / (1+pow(2.7828, -z));
		return sigmoid_z;
	}
	
	float activation_prime(float z){
		float sigmoid_prime_z = activation_function(z) * (1 - activation_function(z));
		return sigmoid_prime_z;
	}
	
	vector<float> network_output(vector<float> input_arr){
		//feed forward output from neural network from reformatted input
		vector<float> z_matrix;
		
		for (int i=0; i < network_length-1; i++){
			float weight = weights[i];
			float bias = biases[i];
			vector<float> total_ls = [0]*bias.size();
			for (int j=0; j < weight.size(); j++){
				float w_vec = weight[j];
				for (int k=0; k < w_vec.size(); k++){
					total_ls += w_vec[k] * input_arr[j];
				}
			}
			
		}
		
		return z_matrix;
	};
	
	void matmult(int mat1[N1][M1], int mat2[N2][M2]){
		int result res[N][M];
		int i, j, l;
		for (i=0; i < N1; i++){
			for (j=0; j < 
		}
	}
	

};



//train the network 
int main() {
	Network connected_net;
	connected_net.print_weights();
	//connected_net.gradient_descent(10, 50, 0.1);
	return 0;
	}















