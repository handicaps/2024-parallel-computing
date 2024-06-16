#include <iostream>
#include <vector>
#include <climits>
#include <queue>
#include <omp.h>

using namespace std;

const int INF = INT_MAX;

// 顶点的结构体，用于堆优化
struct Vertex {
    int id;
    int distance;

    Vertex(int _id, int _distance) : id(_id), distance(_distance) {}

    // 用于堆的比较函数
    bool operator>(const Vertex& other) const {
        return distance > other.distance;
    }
};

int main() {
    int V, E, v0;
    cin >> V >> E >> v0;

    vector<vector<pair<int, int>>> graph(V);
    vector<int> dis(V, INF);
    bool has_change = true;

    for (int i = 0; i < E; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        graph[u].push_back(make_pair(v, w));
        graph[v].push_back(make_pair(u, w));
    }

    dis[v0] = 0;

    while (has_change) {
        has_change = false;

#pragma omp parallel
        {
            priority_queue<Vertex, vector<Vertex>, greater<Vertex>> local_pq;

#pragma omp for
            for (int u = 0; u < V; ++u) {
                if (dis[u] != INF) {
                    local_pq.push(Vertex(u, dis[u]));
                }
            }

#pragma omp for reduction(||:has_change)
            for (int u = 0; u < V; ++u) {
                if (dis[u] != INF) {
                    while (!local_pq.empty()) {
                        Vertex min_vertex = local_pq.top();
                        local_pq.pop();

                        int curr_id = min_vertex.id;
                        int curr_dist = min_vertex.distance;

                        // 如果当前顶点已经更新过，则跳过
                        if (curr_dist > dis[curr_id]) continue;

                        // 并行地对相邻顶点进行松弛操作
                        for (auto& neighbor : graph[curr_id]) {
                            int v = neighbor.first;
                            int weight = neighbor.second;
                            if (dis[curr_id] + weight < dis[v]) {
                                dis[v] = dis[curr_id] + weight;
                                local_pq.push(Vertex(v, dis[v])); // 更新堆中的距离值
#pragma omp atomic write
                                has_change = true; // 发生改变则设置标志
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < V; ++i) {
        if (dis[i] != INF) {
            cout << dis[i] << " ";
        }
        else {
            cout << "INF ";
        }
    }
    cout << endl;

    return 0;
}
