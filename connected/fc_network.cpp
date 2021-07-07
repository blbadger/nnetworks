
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

float random_sample(float mean, float stdev){
	std::random_device rd; 
    std::mt19937 gen(rd()); 
    
    float sample;
	// instance of class std::normal_distribution with specific mean and stddev
	normal_distribution<float> distro(mean, stdev); 

	// get random number with normal distribution using gen as random source
	sample = distro(gen); 
	
	return sample;
	}

class Network{
	vector<int> architecture = {784, 20, 10};
	int network_length = architecture.size();
	vector<vector<float>> biases;
	vector<vector<vector<float>>> weights;

	 // initialize biases
	public: 
		void biases_init(){
			for (int i=0; i < network_length-1; i++){
				vector<float> temp;
				for (int j=0; j < architecture[i]; j++){
					 //obtain number from normal distribution
					float val = random_sample(0, 1); 
					temp.push_back(val);
				}
				biases.push_back(temp);
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
	
	float activation_function(float z){
		float sigmoid_z = 1 / (1+pow(2.7828, -z));
		return sigmoid_z;
	};
	
	float activation_prime(float z){
		float sigmoid_prime_z = activation_function(z) * (1 - activation_function(z));
		return sigmoid_prime_z;
	};
	
	float cost_function_derivative(float output_activations, float y){
		return output_activations - y;
	};
	
	
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















