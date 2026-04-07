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
	ref_count = new int(1); 	

    	for (size_t i = 0; i < total; i++) {
		data[i] = values[i];
    	}
}

Tensor::Tensor(double* data, int* ref_count,
               const std::vector<size_t>& shape)
    : data(data), ref_count(ref_count), shape(shape) {

    (*ref_count)++;
}

Tensor::Tensor(const Tensor& other)
    : data(other.data), ref_count(other.ref_count),
      shape(other.shape), values(other.values) {
    (*ref_count)++;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data), ref_count(other.ref_count),
      shape(std::move(other.shape)), values(std::move(other.values)) {
    other.data = nullptr;
    other.ref_count = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (ref_count) {
	(*ref_count)--;
        if (*ref_count == 0) {
            delete[] data;
            delete ref_count;
        } 
	}
        data = other.data;
        ref_count = other.ref_count;
        shape = other.shape;
        values = other.values;
        (*ref_count)++;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
	    if(ref_count){
        (*ref_count)--;
        if (*ref_count == 0) {
            delete[] data;
            delete ref_count;
        }
	    }
        data = other.data;
        ref_count = other.ref_count;
        shape = std::move(other.shape);
        values = std::move(other.values);
        other.data = nullptr;
        other.ref_count = nullptr;
    }
    return *this;
}

Tensor::~Tensor() {
	if (ref_count) {
	(*ref_count)--; 
	if (*ref_count == 0) {
		delete[] data; 
		delete ref_count; 
	}
	}
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

Tensor Tensor::view(const std::vector<size_t>& new_shape) {

    if (new_shape.size() < 1 || new_shape.size() > 3) {
        throw std::invalid_argument("Dimensiones inválidas");
    }

    size_t total_old = 1;
    for (size_t dim : this->shape) total_old *= dim;

    size_t total_new = 1;
    for (size_t dim : new_shape) total_new *= dim;

    if (total_old != total_new) {
        throw std::invalid_argument("El número de elementos debe coincidir");
    }

    return Tensor(this->data, this->ref_count, new_shape);
}

Tensor Tensor::unsqueeze(const size_t& position) {

    if (position > shape.size()) {
        throw std::invalid_argument("posición inválida");
    }

    if (shape.size() >= 3) {
        throw std::invalid_argument("no puede exceder 3 dimensiones");
    }

    std::vector<size_t> new_shape;

    for (size_t i = 0; i < position; i++) {
        new_shape.push_back(shape[i]);
    }

    new_shape.push_back(1);

    for (size_t i = position; i < shape.size(); i++) {
        new_shape.push_back(shape[i]);
    }

    return Tensor(this->data, this->ref_count, new_shape);
}

Tensor Tensor::concat(const std::vector<Tensor>& tensors, const size_t& dim) {

    if (tensors.empty()) {
        throw std::invalid_argument("no se pueden concatenar 0 tensores");
    }

    size_t num_dims = tensors[0].shape.size();

    for (const auto& t : tensors) {
        if (t.shape.size() != num_dims) {
            throw std::invalid_argument("dimensiones distintas");
        }
    }

    if (dim >= num_dims) {
        throw std::invalid_argument("dimensión inválida");
    }

    for (size_t i = 0; i < num_dims; i++) {
        if (i == dim) continue;
        size_t expected = tensors[0].shape[i];
        for (const auto& t : tensors) {
            if (t.shape[i] != expected) {
                throw std::invalid_argument("dimensiones incompatibles");
            }
        }
    }

    std::vector<size_t> new_shape = tensors[0].shape;
    new_shape[dim] = 0;
    for (const auto& t : tensors) {
        new_shape[dim] += t.shape[dim];
    }

    size_t total = 1;
    for (size_t d : new_shape) total *= d;

    std::vector<double> values;
    values.reserve(total);

    // CASO 1D
    if (num_dims == 1) {
        for (const auto& t : tensors) {
            for (size_t i = 0; i < t.shape[0]; i++) {
                values.push_back(t.data[i]);
            }
        }
    }

    // CASO 2D
    else if (num_dims == 2) {

        if (dim == 0) {
            for (const auto& t : tensors) {
                size_t total_t = t.shape[0] * t.shape[1];
                for (size_t i = 0; i < total_t; i++) {
                    values.push_back(t.data[i]);
                }
            }
        } else { 

            size_t rows = tensors[0].shape[0];

            for (size_t r = 0; r < rows; r++) {
                for (const auto& t : tensors) {
                    size_t cols = t.shape[1];
                    for (size_t c = 0; c < cols; c++) {
                        values.push_back(t.data[r * cols + c]);
                    }
                }
            }
        }
    }

    // CASO 3D
    else if (num_dims == 3) {

        size_t d0 = tensors[0].shape[0];
        size_t d1 = tensors[0].shape[1];
        size_t d2 = tensors[0].shape[2];

        if (dim == 0) {
            for (const auto& t : tensors) {
                size_t total_t = t.shape[0] * t.shape[1] * t.shape[2];
                for (size_t i = 0; i < total_t; i++) {
                    values.push_back(t.data[i]);
                }
            }
        }

        else if (dim == 1) {
            for (size_t i = 0; i < d0; i++) {
                for (const auto& t : tensors) {
                    size_t cur_d1 = t.shape[1];

                    for (size_t j = 0; j < cur_d1; j++) {
                        for (size_t k = 0; k < d2; k++) {
                            values.push_back(
                                t.data[i * (cur_d1 * d2) + j * d2 + k]
                            );
                        }
                    }
                }
            }
        }

        else if (dim == 2) {
            for (size_t i = 0; i < d0; i++) {
                for (size_t j = 0; j < d1; j++) {
                    for (const auto& t : tensors) {
                        size_t cur_d2 = t.shape[2];

                        for (size_t k = 0; k < cur_d2; k++) {
                            values.push_back(
                                t.data[i * (d1 * cur_d2) + j * cur_d2 + k]
                            );
                        }
                    }
                }
            }
        }
    }

    return Tensor(new_shape, std::move(values));
}	

Tensor matmul(const Tensor& a, const Tensor& b) {

    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::invalid_argument("Los tensores deben ser 2D");
    }

    size_t rows_a = a.shape[0];
    size_t cols_a = a.shape[1];
    size_t rows_b = b.shape[0];
    size_t cols_b = b.shape[1];

    if (cols_a != rows_b) {
        throw std::invalid_argument("Dimensiones incompatibles");
    }

    std::vector<size_t> new_shape = {rows_a, cols_b};

    std::vector<double> values;
    values.reserve(rows_a * cols_b);

    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_b; j++) {

            double sum = 0;

            for (size_t k = 0; k < cols_a; k++) {
                sum += a.data[i * cols_a + k] * b.data[k * cols_b + j];
            }

            values.push_back(sum);
        }
    }

    return Tensor(new_shape, std::move(values));
}

Tensor dot(const Tensor& a, const Tensor& b) {

    if (a.shape != b.shape) {
        throw std::invalid_argument("Las dimensiones deben coincidir");
    }

    size_t total = 1;
    for (size_t d : a.shape) total *= d;

    double result = 0.0;
    for (size_t i = 0; i < total; i++) {
        result += a.data[i] * b.data[i];
    }

    return Tensor({1}, {result});
}


Tensor Tensor::apply(const TensorTransform& transform) const {

    size_t total = 1;
    for (size_t d : shape) total *= d;

    std::vector<double> values;
    values.reserve(total);

    for (size_t i = 0; i < total; i++) {
        values.push_back(transform.apply(data[i]));
    }

    return Tensor(shape, std::move(values));
}





