#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
private:
	double* data;
	std::vector<size_t> shape; 
public:
	Tensor(const std::vector<size_t>& shape, const std::vector<double>& values); 
	~Tensor(); 

	static Tensor zeros(const std::vector<size_t>& shape);
	static Tensor ones(const std::vector<size_t>& shape);
	static Tensor random(const std::vector<size_t>& shape, const int& min, const int& max);
	static Tensor arange(const int& start, const int& end);
	
	Tensor operator+(const Tensor& other);
	Tensor operator-(const Tensor& other);
	Tensor operator*(const Tensor& other);
	Tensor operator*(const double& n);

	
	friend Tensor dot(const Tensor& a, const Tensor& b);
	friend Tensor matmul(const Tensor& a, const Tensor& b);
};



#endif
