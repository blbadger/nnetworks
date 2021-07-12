// matrix_multiplication and transposition

#include <iostream>
#include <vector>
#include <cmath>
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
};

vector<vector<float>> matmult(vector<vector<float>> mat1, 
							vector<vector<float>> mat2) {
	vector<vector<float>> res;
	if (mat1[0].size() != mat2.size()){
		cout << "Matrix size error: mat1 row length /= mat2 col length";
		return res;
	}
	for (unsigned int k=0; k < mat1.size(); k++){
		vector<float> temp;
		for (unsigned int i=0; i < mat2[0].size(); i++){
			int sum = 0;
			for (unsigned int j=0; j < mat2.size(); j++){
				sum += mat1[k][j] * mat2[j][i];
			}
			temp.push_back(sum);
		}
		res.push_back(temp);
	}
	return res;
};


vector<vector<float>> transpose(vector<vector<float>> arr) {
	vector<vector<float>> res;
	int rows = arr.size();
	int cols = arr[0].size();
	
	for (int i=0; i < cols; i++){
		vector<float> temp;
		for (int j=0; j < rows; j++) {
			temp.push_back(0.);
		}
		res.push_back(temp);
};
	
	for (int i=0; i < rows; i++){
		for (int j=0; j < cols; j++){
			res[j][i] = arr[i][j];
			}
		}
		
	return res;
};

vector<float> transpose_row(vector<float> arr){
	vector<float> res;
	int size = arr.size();
	for (int i=0; i < size; i++){
		vector<float> ele = {arr[i]};
		res.insert(arr.end(), ele.begin(), ele.end());
		}
	return res;
	}

vector<vector<float>> sigmoid(vector<vector<float>> arr) {
	for (int i=0; i < arr.size(); i++){
		for (int j=0; j < arr[0].size(); j++){
			arr[i][j] = 1 / (1 + std::pow(2.7828, -arr[i][j]));
			}
		}
	return arr;
};
	

vector<float> scalar_mult(vector<float> arr, float scalar){
	vector<float> res;
	for (int i=0; i < arr.size(); i++){
		res.push_back(arr[i] * scalar);
	}
	return res;
};

vector<float> scalar_add(vector<float> arr, float scalar){
	vector<float> res;
	for (int i=0; i < arr.size(); i++){
		res.push_back(arr[i] + scalar);
	}
	return res;
};
	
vector<float> hadamard(vector<float> arr1, vector<float> arr2){
	vector<float> res;
	if (arr1.size() != arr2.size()){
		 cout << "Error: arr1 and arr2 not same size";
		 return res;
	}
	
	for (int i=0; i < arr1.size(); i++){
		res.push_back(arr1[i] * arr2[i]);
	}
	
	return res;
};

vector<vector<float>> matadd(vector<vector<float>> arr1, vector<vector<float>> arr2){
	vector<vector<float>> res;
	
	for (int i=0; i < arr1.size(); i++){
		vector<float> temp;
		for (int j=0; j < arr1[0].size(); j++){
			temp.push_back(arr1[i][j] + arr2[i][j]);
		}
		res.push_back(temp);
	}
	
	return res;
}

/*
int main(){
	 //standard definition: mxn matrix has m rows and n columns
	vector<vector<float>>  mat1 = {{ 1., 3., 5.}};  
								 
	vector<vector<float>> mat2 = {{ 1.}, 
								{ 0. }, 
								{ 17.}}; 
	vector<vector<float>> res;
	res = matmult(mat1, mat2);
	//res = matadd(mat1, transpose(mat2));
	for (int i=0; i < res.size(); i++){
		for (int j=0; j < res[0].size(); j++){
			cout << res[i][j] << " ";
		}
		cout << "\n";
	}
	return 0;
}
*/


