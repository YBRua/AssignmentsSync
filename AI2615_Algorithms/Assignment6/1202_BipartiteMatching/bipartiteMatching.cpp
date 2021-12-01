#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <array>

using namespace std;

int previous[202];
int flow[202];
int gender[202] = {0};
int capacity[202][202] = {0};
array<vector<int>, 502> originalInput;

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
    for (int i = 0; i < 502; ++i)
    {
        previous[i] = -1;
    }

    queue<int> bfsq;
    bool visited[202] = {0};
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
            if (capacity[next][j] == 0) {
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

int maxFlow(array<vector<int>, 502> graph, int size, int source, int sink)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            capacity[i][j] = 0;
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < graph[i].size(); ++j)
        {
            capacity[i][graph[i][j]] = 1;
        }
    }
    while (maxFlowBFS(size, source, sink))
    {
        augment(size, source, sink);
    }

    int maxFlow = 0;
    for (int i = 0; i < graph[source].size(); ++i)
    {
        // flows out of source is flow into source in augGraph
        maxFlow += capacity[graph[source][i]][source];
    }

    return maxFlow;
}

int main()
{
    int numOfCandidates = 0, numOfCounterparts = 0;
    int source, sink;

    // input
    cin >> numOfCandidates >> numOfCounterparts;
    source = numOfCandidates;
    sink = source + 1;
    for (int i = 0; i < numOfCounterparts; ++i)
    {
        int start = -1, end = -1;
        cin >> start >> end;
        originalInput[start].push_back(end);
        originalInput[end].push_back(start);
        capacity[start][end] = 1;
        capacity[end][start] = 1;
    }

    // run a bfs to check the gender of candidates
    bool visited[202];
    queue<int> bfsq;
    for (int i = 0; i < numOfCandidates; ++i)
    {
        previous[i] = -1;
    }
    for (int i = 0; i < numOfCandidates; ++i)
    {
        if (originalInput[i].size() && !visited[i])
        {
            bfsq.push(i);
            while (bfsq.size())
            {
                int next = bfsq.front();
                bfsq.pop();
                gender[next] = previous[next] == -1 ? 1 : -gender[previous[next]];
                visited[next] = true;
                for (int j = 0; j < originalInput[next].size(); ++j)
                {
                    if (!visited[originalInput[next][j]])
                    {
                        previous[originalInput[next][j]] = next;
                        bfsq.push(originalInput[next][j]);
                    }
                }
            }
        }
    }

    // convert to a bipartite
    for (int i = 0; i < numOfCandidates; ++i)
    {
        if (gender[i] == -1)
        {
            for (int j = 0; j < originalInput[i].size(); ++j)
            {
                capacity[i][j] = 0;
            }
            originalInput[i].clear();
            originalInput[i].push_back(sink);
            capacity[i][sink] = 1;
        }
        if (gender[i] == 1)
        {
            originalInput[source].push_back(i);
            capacity[source][i] = 1;
        }
    }

    cout << (numOfCandidates - maxFlow(originalInput, numOfCandidates + 2, source, sink));

    return 0;
}
