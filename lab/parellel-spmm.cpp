#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA 核函数，用于计算稠密矩阵与稀疏矩阵的乘积
__global__ void sparseMatrixMultiply(int M, int N, int P, int K, int* D, int* S_row, int* S_col, int* S_val, int* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        int sum = 0;
        for (int i = 0; i < K; i++) {
            if (S_col[i] == col) {
                sum += D[row * N + S_row[i]] * S_val[i];
            }
        }
        result[row * P + col] = sum;
    }
}

int main() {
    int M, N, P, K;
    std::cin >> M >> N >> P >> K;

    int size_D = M * N * sizeof(int);
    int size_result = M * P * sizeof(int);
    int size_S = K * sizeof(int);

    // 分配主机内存
    std::vector<int> h_D(M * N);
    std::vector<int> h_S_row(K);
    std::vector<int> h_S_col(K);
    std::vector<int> h_S_val(K);
    std::vector<int> h_result(M * P, 0);

    // 输入稠密矩阵D
    for (int i = 0; i < M * N; i++) {
        std::cin >> h_D[i];
    }

    // 输入稀疏矩阵S
    for (int i = 0; i < K; i++) {
        std::cin >> h_S_row[i] >> h_S_col[i] >> h_S_val[i];
    }

    // 分配设备内存
    int *d_D, *d_S_row, *d_S_col, *d_S_val, *d_result;
    cudaMalloc((void**)&d_D, size_D);
    cudaMalloc((void**)&d_S_row, size_S);
    cudaMalloc((void**)&d_S_col, size_S);
    cudaMalloc((void**)&d_S_val, size_S);
    cudaMalloc((void**)&d_result, size_result);

    // 将数据从主机复制到设备
    cudaMemcpy(d_D, h_D.data(), size_D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S_row, h_S_row.data(), size_S, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S_col, h_S_col.data(), size_S, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S_val, h_S_val.data(), size_S, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result.data(), size_result, cudaMemcpyHostToDevice);

    // 定义CUDA网格和块的维度
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用CUDA核函数
    sparseMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(M, N, P, K, d_D, d_S_row, d_S_col, d_S_val, d_result);

    // 将结果从设备复制到主机
    cudaMemcpy(h_result.data(), d_result, size_result, cudaMemcpyDeviceToHost);

    // 输出结果矩阵
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            std::cout << h_result[i * P + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放设备和主机内存
    cudaFree(d_D);
    cudaFree(d_S_row);
    cudaFree(d_S_col);
    cudaFree(d_S_val);
    cudaFree(d_result);

    return 0;
}
