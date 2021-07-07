// matrix_multiplication and transposition

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

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
	}
	
	for (int i=0; i < rows; i++){
		for (int j=0; j < cols; j++){
			res[j][i] = arr[i][j];
			}
		}
		
	return res;
	};

vector<float> sigmoid(vector<float> arr) {
	for (int i=0; i < arr.size(); i++){
		arr[i] = 1 / (1 + std::pow(2.7828, -arr[i]));
		}
	return arr;
	};
	

vector<float> scalar_mult(vector<float> arr, float scalar){
	vector<float> res;
	return res;
	}


int main(){
	 //standard definition: mxn matrix has m rows and n columns
	vector<vector<float>>  mat1 = {{ 1., 3., 5.},
								 { 2., 4., 6.}};  
								 
	vector<vector<float>> mat2 = {{ 1. , 4., 3.}, 
								{ 0. , 5., 1.}, 
								{ 17., 6., 11.}}; 
	vector<vector<int>> res;
	mat2 = transpose(mat2);
	res = matmult(mat1, mat2);
	for (int i=0; i < res.size(); i++){
		for (int j=0; j < res[0].size(); j++){
			cout << res[i][j] << " ";
		}
		cout << "\n";
	}
	return 0;
}

