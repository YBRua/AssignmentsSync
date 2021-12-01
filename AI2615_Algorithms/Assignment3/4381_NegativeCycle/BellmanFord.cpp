#include <iostream>
#include <vector>

using namespace std;

class Edge
{
    public:
    int from;
    int to;
    int weight;

    Edge(int s, int e, int w);
    Edge(){};
};

Edge::Edge(int s, int e, int w)
{
    from = s;
    to = e;
    weight = w;
}

vector<Edge> edges;
const int INFTY = 1 << 30;
vector<int> dist = vector<int>(5000, INFTY);

void BellmanFord(int num_vertices)
{
    for (int i = 0; i < num_vertices - 1; ++i)
    {
        for (int j = 0; j < edges.size(); ++j)
        {
            Edge current = edges[j];
            if (dist[current.to] > dist[current.from] + current.weight)
                dist[current.to] = dist[current.from] + current.weight;
        }
    }
}

int main()
{
    int n, m;
    cin >> n >> m;
    int from, to, weight;

    for (int i = 0; i < m; ++i)
    {
        cin >> from >> to >> weight;
        edges.push_back(Edge(from, to, weight));
    }

    BellmanFord(n);

    for (int j = 0; j < edges.size(); ++j)
    {
        Edge current = edges[j];
        if (dist[current.to] > dist[current.from] + current.weight)
        {
            cout << "Yes";
            return 0;
        }
    }
    cout << "No";

    return 0;
}