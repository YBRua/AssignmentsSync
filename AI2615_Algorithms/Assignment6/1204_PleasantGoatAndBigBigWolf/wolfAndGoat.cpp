#include <iostream>
#include <queue>

#define INFTY 1919810

using namespace std;

int adjMat[1024][1024];

int previous[1024];
int capacity[1024][1024];

void augment(int size, int source, int sink)
{
    int current = sink;
    int pre = previous[current];
    int bottleneck = 1919810;
    while (pre != -1)
    {
        bottleneck = bottleneck < capacity[pre][current] ? bottleneck : capacity[pre][current];
        current = pre;
        pre = previous[current];
    }

    current = sink;
    pre = previous[current];
    while (pre != -1)
    {
        capacity[pre][current] -= bottleneck;
        capacity[current][pre] += bottleneck;
        current = pre;
        pre = previous[current];
    }
}

bool maxFlowBFS(int size, int source, int sink)
{
    for (int i = 0; i < size; ++i)
    {
        previous[i] = -1;
    }

    queue<int> bfsq;
    bool visited[1024] = {0};
    bfsq.push(source);
    while (bfsq.size())
    {
        int next = bfsq.front();
        bfsq.pop();
        visited[next] = true;

        if (next == sink)
            return true;

        for (int j = 0; j < size; ++j)
        {
            if (capacity[next][j] == 0)
            {
                continue;
            }
            if (!visited[j])
            {
                previous[j] = next;
                bfsq.push(j);
            }
        }
    }
    return false;
}

int maxFlow(int size, int source, int sink)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            capacity[i][j] = adjMat[i][j];
        }
    }
    while (maxFlowBFS(size, source, sink))
    {
        augment(size, source, sink);
    }

    int maxFlow = 0;
    for (int i = 0; i < size; ++i)
    {
        // flows out of source is flow into source in augGraph
        maxFlow += capacity[i][source];
    }

    return maxFlow;
}

inline int loc2Index(int x, int y, int m)
{
    return x * m + y;
}

inline void addEdge(int x1, int y1, int x2, int y2, int n, int m)
{
    if (
        loc2Index(x2, y2, n) < 0 || x1 < 0 || x2 < 0 || y1 < 0 || y2 < 0 || x1 >= n || x2 >= n || y1 >= m || y2 >= m)
    {
        return;
    }
    adjMat[loc2Index(x1, y1, m)][loc2Index(x2, y2, m)] = 1;
    adjMat[loc2Index(x2, y2, m)][loc2Index(x1, y1, m)] = 1;
}

inline void addEdge(int s, int t)
{
    if (s < 0 || t < 0)
        return;
    adjMat[s][t] = 1;
    adjMat[t][s] = 1;
}

int main()
{
    int n, m;
    cin >> n >> m;
    // init adjacent matrix
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            addEdge(i, j, i + 1, j, n, m);
            addEdge(i, j, i, j + 1, n, m);
            addEdge(i, j, i - 1, j, n, m);
            addEdge(i, j, i, j - 1, n, m);
        }
    }

    // read input
    // convert graph
    int source = m * n;
    int sink = source + 1;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            int faction;
            cin >> faction;
            if (faction == 1)
            {
                adjMat[source][loc2Index(i, j, m)] = INFTY;
            }
            if (faction == 2)
            {
                adjMat[loc2Index(i, j, m)][sink] = INFTY;
            }
        }
    }

    // for (int i = 0; i < m * n + 2; ++i)
    // {
    //     cout << "Node " << i << ' ';
    //     for (int j = 0; j < m * n + 2; ++j)
    //     {
    //         cout << adjMat[i][j] << ' ';
    //     }
    //     cout << endl;
    // }

    cout << maxFlow(m * n + 2, source, sink);

    return 0;
}
