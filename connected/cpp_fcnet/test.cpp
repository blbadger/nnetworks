#include <iostream>
#include <iomanip>
#include <random>
#include "matrix_operations.cpp"

using namespace std;

vector<float> activation_function(vector<float> z_array){
	vector<float> sigmoid_arr = sigmoid(z_array); // from matrix_operations.cpp
	return sigmoid_arr;
};

vector<float> activation_prime(vector<float> z){
	vector<float> neg_z = scalar_mult(z, -1);
	vector<float> sigmoid_prime_z = hadamard(activation_function(z), scalar_add(neg_z, 1));
	return sigmoid_prime_z;
};

int main(){
	vector<float> arr = {1, 2, 3};
	vector<float> arr2 = {5, 6, 7};
	float scalar = 5;
	vector<float> fin;
    fin = activation_prime(arr);
	for (auto u: fin) cout << u << " ";
}
