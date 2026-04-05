#include <iostream>
#include "Tensor.h"

Tensor::Tensor(int v) : value(v) {}

void Tensor::print() const {
    std::cout << "Tensor cambiado: " << value << std::endl;
}
