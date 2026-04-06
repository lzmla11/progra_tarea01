#include <iostream>
#include "../include/Tensor.h"

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values) 
	: shape(shape), values(values) {
	
	if (shape.size() < 1 || shape.size() > 3) {
        	throw std::invalid_argument("El tensor debe tener entre 1 y 3 dimensiones");
    	}

    	size_t total = 1;
    	for (size_t dim : shape) {
		total *= dim;
    	}

    	if (values.size() != total) {
		throw std::invalid_argument("Cantidad de valores incorrecta");
    	}

	data = new double[total];

    	for (size_t i = 0; i < total; i++) {
		data[i] = values[i];
    	}
}

Tensor::~Tensor() {
	delete data[]; 
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }

    std::vector<double> values(total, 0.0);

    return Tensor(shape, values);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }

    std::vector<double> values(total, 1.0);

    return Tensor(shape, values);
}

Tensor Tensor::arange(const int& start, const int& end) {

    size_t total = end - start;

    std::vector<double> values;
    values.reserve(total);

    for (int i = start; i < end; i++) {
        values.push_back(i);
    }

    return Tensor({total}, values);
}

Tensor Tensor::random(const std::vector<size_t>& shape,
                      const int& min,
                      const int& max) {

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }

    std::vector<double> values;
    values.reserve(total);

    for (size_t i = 0; i < total; i++) {
        values.push_back(rand() % (max - min) + min);
    }

    return Tensor(shape, values);
}

Tensor Tensor::operator+(const Tensor& other) {

    if (this->shape != other.shape) {
        throw std::invalid_argument("Dimensiones incompatibles");
    }

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }

    std::vector<double> values;
    values.reserve(total);

    for (size_t i = 0; i < total; i++) {
        values.push_back(this->data[i] + other.data[i]);
    }

    return Tensor(this->shape, values);
}


Tensor Tensor::operator-(const Tensor& other) {

    if (this->shape != other.shape) {
        throw std::invalid_argument("Dimensiones incompatibles");
    }

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }

    std::vector<double> values;
    values.reserve(total);

    for (size_t i = 0; i < total; i++) {
        values.push_back(this->data[i] - other.data[i]);
    }

    return Tensor(this->shape, values);
}

Tensor Tensor::operator*(const Tensor& other) {

    if (this->shape != other.shape) {
        throw std::invalid_argument("Dimensiones incompatibles");
    }

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }

    std::vector<double> values;
    values.reserve(total);

    for (size_t i = 0; i < total; i++) {
        values.push_back(this->data[i] * other.data[i]);
    }

    return Tensor(this->shape, values);
}

Tensor Tensor::operator*(const double& n) {

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }

    std::vector<double> values;
    values.reserve(total);

    for (size_t i = 0; i < total; i++) {
        values.push_back(this->data[i] * n);
    }

    return Tensor(this->shape, values);
}
