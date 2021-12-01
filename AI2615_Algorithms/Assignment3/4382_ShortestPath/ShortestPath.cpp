#include <iostream>
#include <vector>
#include <queue>

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

vector<Edge> adjList[150000];

class Vertex
{
    public:
    int id;
    int distance;

    Vertex(int n, int d){id = n; distance = d;};
    Vertex(){};
    friend bool operator>(Vertex a, Vertex b){return a.distance > b.distance;};
    friend bool operator<(Vertex a, Vertex b){return a.distance < b.distance;};
};

int Dijkstra(int start, int end, int num_vertices)
{
    const int INFTY = 1 << 30;
    priority_queue<Vertex, vector<Vertex>, greater<Vertex>> Q;
    vector<int> distance;
    // not needed. 
    // vector<int> previous;

    for (int i = 0; i < num_vertices; ++i)
    {
        distance.push_back(INFTY);
        // previous.push_back(-1);
    }
    distance[start] = 0;
    Q.push(Vertex(start, 0));

    while (!Q.empty())
    {
        Vertex currentNode = Q.top();
        Q.pop();
        for (int i = 0; i < adjList[currentNode.id].size(); ++i)
        {
            Edge currentEdge = adjList[currentNode.id][i];
            if (distance[currentEdge.to] > distance[currentEdge.from] + currentEdge.weight)
            {
                distance[currentEdge.to] = distance[currentEdge.from] + currentEdge.weight;
                // not needed
                // previous[currentEdge.to] = currentEdge.from;
                Q.push(Vertex(currentEdge.to, distance[currentEdge.to]));
            }
        }
    }

    return distance[end];
}


int main()
{
    int n, m, start, end;
    int from, to, weight;
    cin >> n >> m >> start >> end;
    --start;
    --end;

    for (int i = 0; i < m; ++i)
    {
        cin >> from >> to >> weight;
        --from;
        --to;
        adjList[from].push_back(Edge(from, to, weight));
        adjList[to].push_back(Edge(to, from, weight));
    }

    cout << Dijkstra(start, end, n);

    return 0;
}