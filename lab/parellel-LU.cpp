#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <omp.h>

//方阵
std::vector<std::vector<int>> A;
int N;
std::vector<std::vector<int>> L, U;

//按公式计算，L、U矩阵中i行j列的累加和（K个累加和）
int sum_i_j_K(int i, int j, int K) {
    int res = 0;
    for (int k = 0; k < K; k++) {
        res += L[i][k] * U[k][j];
    }
    return res;
}

int main(int argc, char *argv[]) {
    int N;
    if (std::scanf("%d", &N)) {}

    A.resize(N);
    for (int i = 0; i < N; i++) {
        A[i].resize(N);
        for (int j = 0; j < N; j++) {
            if (std::scanf("%d", &A[i][j])) {}
        }
    }
    //生成和初始化 L、U矩阵
    L.resize(N, std::vector<int>(N, 0));
    U.resize(N, std::vector<int>(N, 0));

    //设置线程数
    int max_threads = std::min(omp_get_max_threads(), std::max(3, int(N / 200)));
    omp_set_num_threads(max_threads);

    //计算L、U矩阵
    for (int i = 0; i < N; i++) {
        U[i][i] = A[i][i] - sum_i_j_K(i, i, i);
        L[i][i] = 1;

#pragma omp parallel for
        for (int j = i + 1; j < N; j++) {
            //按照递推公式进行计算
            int sum_u = sum_i_j_K(i, j, i);
            int sum_l = sum_i_j_K(j, i, i);
            
#pragma omp critical
            {
                U[i][j] = A[i][j] - sum_u;
                L[j][i] = (A[j][i] - sum_l) / U[i][i];
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::fixed << std::setprecision(0) << L[i][j] << " ";
        }
        std::cout << std::endl;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::fixed << std::setprecision(0) << U[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
