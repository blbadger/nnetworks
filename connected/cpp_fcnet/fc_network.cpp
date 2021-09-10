
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
	vector<float> architecture = {4, 3, 2};

	int network_length = architecture.size();
	
	//vector<vector<vector<float>>> weights = {{{3, 2, 4, 1}, 
										  //{2, 1, 0, -1},
										  //{0, 3, -1, 1}},
										 //{{-1, 2, 1},
										  //{0.5, 1, 0}}};
										  
	//vector<vector<vector<float>>> biases = {{{2},
											 //{0},
											 //{-1}},
											//{{1},
											 //{-1}}};
	
	vector<vector<vector<float>>> biases;
	vector<vector<vector<float>>> weights;
	vector<vector<vector<float>>> z_vectors;	
	
	float learning_rate = 0.01;

	 // initialize biases
	public: 
		void biases_init(){
			
			for (int i=1; i < network_length; i++){
				vector<float> temp;
				vector<vector<float>> temp2;
				for (int j=0; j < architecture[i]; j++){
					 //obtain number from normal distribution
					float val = random_sample(0, 1); 
					temp.push_back(val);
					temp2.push_back(temp);
					temp.clear();
					
				}
				biases.push_back(temp2);
			}
			return;
		}
	
		void print_biases(){
			cout << "{";
			cout << "\n";
			for (auto u: biases){
				cout << "{";
				for (auto v: u){
					cout << "{";
					for (auto w: v) cout << w << " ";
					cout << "}, ";
				}
				cout << "}";
				cout << "\n";
			}
			cout << "}";
			cout << "\n" << "\n";
		}
	
		//initialize weights
		void weights_init(){
			for (int i=1; i < network_length; i++){
				vector<vector<float>> layer;
				for (int j=0; j < architecture[i]; j++){
					vector<float> temp;
					for (int k=0; k < architecture[i-1]; k++){
						float val = random_sample(0, 1);
						temp.push_back(val);
					}
					layer.push_back(temp);
				}
				weights.push_back(layer);
			}
			return;
		}
	
		void print_weights(){
			cout << "{";
			cout << "\n";
			for (auto u: weights){
				cout << "{";
				for (auto v: u){
					cout << "{";
					for (auto q:v){
						cout << q << " ";
					}
					cout << "}, ";
				}
				cout << "}";
				cout << "\n";
			}
			cout << "}";
			cout << "\n" << "\n";
		}
	
	vector<vector<float>> activation_function(vector<vector<float>> z_array){
		
			vector<vector<float>> sigmoid_arr = sigmoid(z_array); 
			return sigmoid_arr;
		};


	vector<vector<float>> activation_prime(vector<vector<float>> z){
		
		vector<vector<float>> neg_z = scalar_mult(z, -1);
		vector<vector<float>> sigmoid_prime_z = hadamard(activation_function(z), scalar_add(neg_z, 1));
		return sigmoid_prime_z;
	};
	

	vector<vector<float>> cost_function_derivative(vector<vector<float>> output_activations, vector<vector<float>> y){
		
		vector<vector<float>> neg_y = scalar_mult(y, -1);
		return matadd(output_activations, neg_y);
	};
	
	
	vector<vector<float>> network_output(vector<vector<float>> output){
		
		for (int i=0; i < int(architecture.size()) - 1; i++){
			vector<vector<float>> weight_arr = weights[i];
			vector<vector<float>> biases_arr = biases[i];
			vector<vector<float>> transposed_weight = transpose(weight_arr);
			
			vector<vector<float>> z_vec = matmult(weight_arr, output);
			
			vector<vector<float>> activations = matadd(z_vec, biases_arr);
			output = activation_function(activations);
			
		}
		return output;
	}
	
	vector<vector<vector<float>>> forward(vector<vector<float>> output){
		
		vector<vector<vector<float>>> activations_arr;
		activations_arr.push_back(output);
		
		for (int i=1; i < architecture.size(); i++){
			vector<vector<float>> weight_arr = weights[i-1];
			vector<vector<float>> biases_arr = biases[i-1];
			vector<vector<float>> transposed_weight= transpose(weight_arr); 

			vector<vector<float>> z_vec = matmult(weight_arr, output);
			vector<vector<float>> activations = matadd(z_vec, biases_arr);
			
			output = activation_function(activations);
			activations_arr.push_back(output);
				
			z_vectors.push_back(z_vec);
		}

		return activations_arr;
	}

	
	void backpropegate(vector<vector<float>> output, vector<vector<float>> classification){
		
		// compute output error
		vector<vector<vector<float>>> activations_arr = forward(output);
		
		vector<vector<float>> last_acts = activations_arr[activations_arr.size()-1];
		vector<vector<float>> last_vecs = z_vectors[z_vectors.size()-1];
		vector<vector<float>> last_prime = activation_prime(last_vecs);
		
		vector<vector<float>> error = hadamard(cost_function_derivative(last_acts, classification), last_prime);
		
		// initialize partial derivative arrays
		vector<vector<vector<float>>> dc_db;
		vector<vector<vector<float>>> dc_dw;
			
		// Partial derivatives of the last layer wrt error
		dc_db.push_back(error);
		dc_dw.push_back(matmult(error, transpose(activations_arr[activations_arr.size()-2])));
		
			 
		// backpropegate (do not include input layer specified by architecture[0])
		for (int i = architecture.size() - 2; i >= 1; i--){

			vector<vector<float>> act = activation_prime(z_vectors[i-1]);
			vector<vector<float>> w_err = matmult(transpose(weights[i]), error);
			vector<vector<float>> error =  hadamard(w_err, act);
			
			//update partial derivatives with error
			dc_db.push_back(error);
			dc_dw.push_back(matmult(error, transpose(activations_arr[i-1])));
		}
		
		// compile array of partial derivatives
		vector<vector<vector<float>>> partial_db = reverse(dc_db);
		dc_dw = reverse(dc_dw);
		vector<vector<vector<float>>> partial_dw;
		
		for (int i=0; i < int(dc_dw.size()); i++){
			partial_dw.push_back(dc_dw[i]);
		}

		// gradient descent: update weights and biases
		float lr = learning_rate;
		
		for (int i=0; i < int(weights.size()); i++){
			vector<vector<float>> direction = scalar_mult(scalar_mult(partial_dw[i], lr), -1);
			weights[i] = matadd(weights[i], direction);
		}
		
		for (int i=0; i < int(biases.size()); i++){
			vector<vector<float>> direction = scalar_mult(scalar_mult(partial_db[i], lr), -1);
			biases[i] = matadd(biases[i], direction);
		}
			
		//vector<vector<vector<float>>> fin = weights;

		//for (int i=0; i < int(fin.size()); i++){
			//for (int j=0; j < int(fin[i].size()); j++){
				//for (int k=0; k < int(fin[i][j].size()); k++){
					//cout << fin[i][j][k] << " ";
				//}
				//cout << "\n";
			//}
			//cout << "\n" << "\n";
		//}
	}

};


//train the network 
int main() {
	vector<vector<float>> output = {{1},
									{0},
									{-1},
									{0.1}};
								
	vector<vector<float>> classification = {{0.}, {1.}};	
	
	
	Network connected_net;
	connected_net.biases_init();
	connected_net.weights_init();
	
	// check initial weights and biases
	connected_net.print_biases();
	connected_net.print_weights();
	connected_net.backpropegate(output, classification);
	
	// check modified weights and biases
	connected_net.print_biases();
	connected_net.print_weights();
	return 0;
}
















