#include "../include/Tensor.h"
#include <iostream>

int main() {
    srand(time(nullptr));

    // Constructor principal y tensores predefinidos
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor Z = Tensor::zeros({2, 3});
    Tensor O = Tensor::ones({3, 3});
    Tensor R = Tensor::random({2, 2}, 0, 10);
    Tensor D = Tensor::arange(0, 6);

    // Rule of Five
    Tensor copia(A);
    Tensor movido(std::move(Tensor::ones({2, 2})));
    Tensor asignado = Tensor::zeros({2, 3});
    asignado = A;
    Tensor asignado_move = Tensor::zeros({2, 3});
    asignado_move = std::move(copia);

    // Operadores
    Tensor B({2, 3}, {6, 5, 4, 3, 2, 1});
    Tensor C = A + B;
    Tensor E = A - B;
    Tensor F = A * B;
    Tensor G = A * 2.0;

    // view y unsqueeze
    Tensor V  = Tensor::arange(0, 12).view({3, 4});
    Tensor U  = Tensor::arange(0, 3);
    Tensor U1 = U.unsqueeze(0);  // {1, 3}
    Tensor U2 = U.unsqueeze(1);  // {3, 1}

    // concat
    Tensor CA = Tensor::ones({2, 3});
    Tensor CB = Tensor::zeros({2, 3});
    Tensor CC = Tensor::concat({CA, CB}, 0);  // {4, 3}
    Tensor CD = Tensor::concat({CA, CB}, 1);  // {2, 6}

    // dot y matmul
    Tensor da({1, 3}, {1, 2, 3});
    Tensor db({1, 3}, {4, 5, 6});
    Tensor dr = dot(da, db);
    Tensor MA({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor MB({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor MC = matmul(MA, MB);

    // Red neuronal (Seccion 10)
    ReLU relu;
    Sigmoid sigmoid;

    // Paso 1: entrada 1000x20x20
    Tensor input = Tensor::random({1000, 20, 20}, 0, 10);
    // Paso 2: view -> 1000x400
    Tensor flat = input.view({1000, 400});
    // Paso 3: matmul con W1 -> 1000x100
    Tensor W1 = Tensor::random({400, 100}, 0, 5);
    Tensor layer1 = matmul(flat, W1);
    // Paso 4: suma bias b1
    Tensor b1 = Tensor::random({1000, 100}, 0, 3);
    layer1 = layer1 + b1;
    // Paso 5: ReLU
    layer1 = layer1.apply(relu);
    // Paso 6: matmul con W2 -> 1000x10
    Tensor W2 = Tensor::random({100, 10}, 0, 5);
    Tensor layer2 = matmul(layer1, W2);
    // Paso 7: suma bias b2
    Tensor b2 = Tensor::random({1000, 10}, 0, 3);
    layer2 = layer2 + b2;
    // Paso 8: Sigmoid
    Tensor output = layer2.apply(sigmoid);

    std::cout << "shape: 1000 x 10" << std::endl;

    return 0;
}
