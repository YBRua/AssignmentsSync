// ACKNOWLEDGEMENT: this solution is inspired by some random programming dude on the Internet.
/**
 * Each customer is a node.
 * For the first customer of each farm, add an edge from source to customer with capacity = numOfPigs;
 * For later customers, add an edge from previous customer to him with capacity infty
 * For all customers, add an edge to sink with capacity = requirement;
 */
#include <iostream>
#include <queue>

#define INFTY 1919810

using namespace std;

int farms[1024];
int ntr[1024]; // -1 if no customers visited, else index of previous customer
int piggyGraph[105][105];

int previous[105];
int capacity[105][105] = {0};

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
    for (int i = 0; i < 105; ++i)
    {
        previous[i] = -1;
    }

    queue<int> bfsq;
    bool visited[105] = {0};
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
            capacity[i][j] = piggyGraph[i][j];
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

int main()
{
    for (int i = 0; i < 105; ++i)
    {
        for (int j = 0; j < 1 - 5; ++j)
        {
            piggyGraph[i][j] = 0;
        }
    }
    for (int i = 0; i < 1024; ++i)
    {
        ntr[i] = -1;
    }
    int numOfFarms, numOfCustomers;

    // input
    cin >> numOfFarms >> numOfCustomers;
    int source = 0;
    int sink = numOfCustomers + 1;
    // init farms
    for (int i = 1; i <= numOfFarms; ++i)
    {
        int numOfPigs;
        cin >> numOfPigs;
        farms[i] = numOfPigs;
    }
    // init graph
    for (int currentCustomer = 1; currentCustomer <= numOfCustomers; ++currentCustomer)
    {
        // i = 1 : numOfCustomers
        int numOfKeys;
        cin >> numOfKeys;
        for (int k = 0; k < numOfKeys; ++k)
        {
            int key;
            cin >> key;
            if (ntr[key] == -1)
            {
                // first customer
                ntr[key] = currentCustomer;
                piggyGraph[source][currentCustomer] += farms[key];
            }
            else
            {
                // following customers
                int previousCustomer = ntr[key];
                ntr[key] = currentCustomer;
                piggyGraph[previousCustomer][currentCustomer] = INFTY;
            }
        }
        int requirement;
        cin >> requirement;
        // link customers to sinks
        piggyGraph[currentCustomer][sink] = requirement;
    }

    cout << maxFlow(numOfCustomers + 2, source, sink);

    return 0;
}