
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
#include "matrix_operations.cpp"

using namespace std;

float random_sample(float mean, float stdev){
	std::random_device rd; 
    std::mt19937 gen(rd()); 
    
    float sample;
	// instance of class std::normal_distribution with specific mean and stddev
	normal_distribution<float> distro(mean, stdev); 

	// get random number with normal distribution using gen as random source
	sample = distro(gen); 
	
	return sample;
};

class Network{
	vector<int> architecture = {784, 20, 10};
	int network_length = architecture.size();
	vector<vector<vector<float>>> biases;
	vector<vector<vector<float>>> weights;
	float learning_rate = 0.01;

	 // initialize biases
	public: 
		void biases_init(){
			for (int i=0; i < network_length-1; i++){
				vector<float> temp;
				vector<vector<float>> temp2;
				for (int j=0; j < architecture[i]; j++){
					 //obtain number from normal distribution
					float val = random_sample(0, 1); 
					temp.push_back(val);
				}
				temp2.push_back(temp);
				biases.push_back(temp2);
			}
			return;
		}
	
		//void print_biases(){
			//cout << biases.size();
			//for (auto u: biases){
				//for (auto v: u){
					//cout << v;
				//}
			//}
		//}
	
	 //initialize weights
		void weights_init(){
			for (int i=0; i < network_length-1; i++){
				vector<vector<float>> layer;
				for (int j=0; j < architecture[i]; j++){
					vector<float> temp;
					for (int k=0; k < architecture[i]; k++){
						float val = random_sample(0, 1);
						temp.push_back(val);
					}
					layer.push_back(temp);
				}
				weights.push_back(layer);
			}
			return;
		}
	
		//void print_weights(){
			//for (auto u: weights){
				//for (auto v: u){
					//for (auto q:v){
						//cout << q;
					//}
				//}
			//}
		//}
	
	vector<vector<float>> activation_function(vector<vector<float>> z_array){
			vector<vector<float>> sigmoid_arr = sigmoid(z_array); 
			return sigmoid_arr;
		};


	vector<vector<float>> activation_prime(vector<vector<float>> z){
		vector<vector<float>> neg_z = scalar_mult(z, -1);
		
		vector<vector<float>> sigmoid_prime_z = hadamard(activation_function(z), scalar_add(neg_z, 1));
		return sigmoid_prime_z;
	};

	float cost_function_derivative(float output_activations, float y){
		return output_activations - y;
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
	
	void backpropegate(vector<vector<float>> output, vector<vector<float>> classification){
		vector<vector<vector<float>>> activations_arr;
		vector<vector<vector<float>>> z_vectors;
		
		for (int i=0; i < architecture.size() - 1; i++){
			vector<vector<float>> weight_arr = weights[i];
			vector<vector<float>> biases_arr = biases[i];
			vector<vector<float>> transposed_output = transpose(output);

			vector<vector<float>> z_vec = matmult(weight_arr, transposed_output);
			vector<vector<float>> activations = matadd(transpose(z_vec), biases_arr);
			
			output = activation_function(activations);
			activations_arr.push_back(output);
			z_vectors.push_back(activations);
		}
		
		// compute output error
		vector<vector<float>> error = cost_function_derivative(activations_arr[activations_arr.size()-1], classification) \
		 * activation_prime(z_vectors[z_vectors.size()-1]);
		 
		// initialize partial derivative arrays
		vector<vector<vector<float>>> dc_db;
		vector<vector<vector<float>>> dc_dw;
		
		// Partial derivatives of the last layer wrt error
		dc_db.push_back(error)
		dc_dw.push_back(matmult(error, transpose(activations[activations.size()-2]));
		
		
		// backpropegate
		for (int i=architecture.size() - 1; i > 0; i--){
			vector<vector<float>> activation = activation_function(z_vec[i]);
			vector<vector<float>> error = matmult(weights[i], error) * activation;
			
			//update partial derivatives with error
			dc_db.push_back(error);
			dc_dw.push_back(matmult(error, transpose(activations[i - 1]);
		}
		
		// update weights and biases with gradient
		
		dc_db = reverse(dc_db);
		dc_dw = reverse(dc_dw);
	
		// gradient descent
		float lr = learning_rate;
		for (int i=0; i < weights.size(); i++){
			for (int j=0; j < weights[i].size(); j++){
				for (int k=0; k < weights[i][j].size(); k++){
					weights[i][j][k] = weights[i][j][k] - lr * dw_db[i][j][k];
				}
			}
		}
		
		for (int i=0; i < biases.size(); i++){
			for (int j=0; j < biases[i].size(); j++){
				for (int k=0; k < biases[i][j].size(); k++){
					biases[i][j][k] = biases[i][j][k] - lr * dc_db[i][j][k];
				}
			}
		}

};


//train the network 
int main() {
	Network connected_net;
	connected_net.biases_init();
	connected_net.weights_init();
	//connected_net.print_weights();
	//connected_net.gradient_descent(10, 50, 0.1);
	return 0;
	}















