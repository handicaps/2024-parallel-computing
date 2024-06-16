#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <limits>
#include <iomanip>

using namespace std;

// 生成随机数
double random_double(double min, double max) {
    static std::mt19937 gen(std::random_device{}()); // 随机数生成
    std::uniform_real_distribution<> dis(min, max);  // 产生范围内均匀取样的数据
    return dis(gen);
}

// 初始化中心点
std::vector<std::vector<double>> initial_value(int n, int k, const std::vector<std::vector<double>>& data) {
    std::vector<double> minJ(n, std::numeric_limits<double>::max());
    std::vector<double> maxJ(n, std::numeric_limits<double>::lowest());

    for (const auto& row : data) {
        for (int i = 0; i < n; ++i) {
            if (row[i] < minJ[i]) minJ[i] = row[i];
            if (row[i] > maxJ[i]) maxJ[i] = row[i];
        }
    }
    std::vector<std::vector<double>> centroids(k, std::vector<double>(n));
    for (auto& centroid : centroids) {
        for (int i = 0; i < n; ++i) {
            centroid[i] = minJ[i] + (maxJ[i] - minJ[i]) * random_double(0, 1);
        }
    }
    return centroids;
}

// 计算欧式距离
double point_distance(const std::vector<double>& point, const std::vector<double>& centroid) {
    double dist = 0;
    for (int j = 0; j < point.size(); ++j) {
        dist += std::pow(point[j] - centroid[j], 2);
    }
    return std::sqrt(dist);
}

int closest_centroid(const std::vector<double>& point, const std::vector<std::vector<double>>& centroids) {
    int min_index = 0;
    double min_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < centroids.size(); ++i) {
        double dist = point_distance(point, centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }
    return min_index;
}

// 计算所有点到其对应的聚类中心的距离之和
double compute_total_distance(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& centroids, const std::vector<int>& assignments) {
    double total_distance = 0.0;
    for (int i = 0; i < data.size(); ++i) {
        total_distance += point_distance(data[i], centroids[assignments[i]]);
    }
    return total_distance;
}

void kmeans_MPI(int k, std::vector<std::vector<double>>& data, int iters, std::vector<std::vector<double>> centroids) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 获取进程号
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int N = data.size();
    int M = data[0].size();
    int local_N = N / size;

    std::vector<std::vector<double>> local_data(local_N, vector<double>(M));

    // Correctly prepare buffer for MPI_Scatter and MPI_Gather
    std::vector<double> data_flat, local_data_flat(local_N * M);
    if (rank == 0) {
        data_flat.resize(N * M);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                data_flat[i * M + j] = data[i][j];
            }
        }
    }

    MPI_Scatter(data_flat.data(), local_N * M, MPI_DOUBLE, local_data_flat.data(), local_N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < M; ++j) {
            local_data[i][j] = local_data_flat[i * M + j];
        }
    }

    // Broadcast centroids
    std::vector<double> centroids_flat(k * M);
    if (rank == 0) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < M; ++j) {
                centroids_flat[i * M + j] = centroids[i][j];
            }
        }
    }
    MPI_Bcast(centroids_flat.data(), k * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < M; ++j) {
            centroids[i][j] = centroids_flat[i * M + j];
        }
    }

    vector<int> all_labels(N);
    vector<int> labels(local_N);
    for (int iter = 0; iter < iters; ++iter) {
        //vector<int> labels(local_N);
        for (int i = 0; i < local_N; ++i) {
            labels[i] = closest_centroid(local_data[i], centroids);
        }

        MPI_Gather(labels.data(), local_N, MPI_INT, all_labels.data(), local_N, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            vector<int> counts(k, 0);
            vector<vector<double>> sums(k, vector<double>(M, 0.0));
            for (int i = 0; i < N; i++) {
                int cluster = all_labels[i];
                for (int j = 0; j < M; j++) {
                    sums[cluster][j] += data[i][j];
                }
                counts[cluster]++;
            }
            for (int i = 0; i < k; ++i) {
                if (counts[i] == 0) continue;
                for (int j = 0; j < M; ++j) {
                    sums[i][j] /= counts[i];
                }
            }
            centroids = sums;

            // Flatten centroids for broadcasting
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < M; ++j) {
                    centroids_flat[i * M + j] = centroids[i][j];
                }
            }
        }

        MPI_Bcast(centroids_flat.data(), k * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < M; ++j) {
                centroids[i][j] = centroids_flat[i * M + j];
            }
        }
    }

    double local_distance = compute_total_distance(local_data, centroids, labels);
    double total_distance;
    MPI_Reduce(&local_distance, &total_distance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(2) << total_distance << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 获取进程号
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 获取总进程数

    const int iters = 100;
    int N, M, K;
    std::vector<std::vector<double>> data;

    if (rank == 0) {
        // 读取输入数据
        std::cin >> N >> M >> K;
        data.resize(N, std::vector<double>(M));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                std::cin >> data[i][j];
            }
        }
    }

    // 广播数据大小给所有进程
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        data.resize(N, std::vector<double>(M)); // 分配空间用于广播数据
    }

    std::vector<std::vector<double>> centroids;
    if (rank == 0) {
        centroids = initial_value(M, K, data);
    }
    else {
        centroids.resize(K, std::vector<double>(M)); // 分配空间用于广播centroids
    }

    kmeans_MPI(K, data, iters, centroids);

    MPI_Finalize();
    return 0;
}
