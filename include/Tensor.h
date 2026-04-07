#ifndef TENSOR_H
#define TENSOR_H

class TensorTransform {
public:
    virtual double apply(double x) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    double apply(double x) const override {
        return x > 0 ? x : 0;
    }
};

class Sigmoid : public TensorTransform {
public:
    double apply(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

class Tensor {
private:
	double* data;
	int* ref_count; 
	std::vector<size_t> shape; 
public:
	Tensor(const std::vector<size_t>& shape, const std::vector<double>& values); 
	Tensor(double* data, int* ref_count, const std::vector<size_t>& shape);
	~Tensor(); 

	static Tensor zeros(const std::vector<size_t>& shape);
	static Tensor ones(const std::vector<size_t>& shape);
	static Tensor random(const std::vector<size_t>& shape, const int& min, const int& max);
	static Tensor arange(const int& start, const int& end);
	
	Tensor operator+(const Tensor& other);
	Tensor operator-(const Tensor& other);
	Tensor operator*(const Tensor& other);
	Tensor operator*(const double& n);

	Tensor view(const std::vector<size_t>& new_shape);
	Tensor unsqueeze(const size_t& position);
	static Tensor concat(const std::vector<Tensor>& tensors, const size_t& dim);

	friend Tensor dot(const Tensor& a, const Tensor& b);
	friend Tensor matmul(const Tensor& a, const Tensor& b);

	Tensor apply(const TensorTransform& tensorTransform);
};



#endif
