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
    friend bool operator<(Edge a, Edge b);
};

Edge::Edge(int s, int e, int w)
{
    from = s;
    to = e;
    weight = w;
}

bool operator<(Edge a, Edge b)
{
    return a.weight < b.weight;
}


Edge edges[300000];
int connectivity[150000];


template <typename T>
void Merge(T arr[], int left, int mid, int right)
{
    int leftLength = mid - left + 1;
    int rightLength = right - mid;
    T* tempLeft = new T[leftLength];
    T* tempRight = new T[rightLength];

    for(int i = 0; i < leftLength; ++i)
        tempLeft[i] = arr[left + i];
    for(int i = 0; i < rightLength; ++i)
        tempRight[i] = arr[mid + 1 + i];

    int i = 0;
    int j = 0;
    int k = left;
    while(i < leftLength && j < rightLength)
    {
        if(tempLeft[i] < tempRight[j])
        {
            arr[k] = tempLeft[i];
            ++i;
        }
        else
        {
            arr[k] = tempRight[j];
            ++j;
        }
        ++k;
    }
    while(i < leftLength)
    {
        arr[k] = tempLeft[i];
        ++k;
        ++i;
    }
    while(j < rightLength)
    {
        arr[k] = tempRight[j];
        k++;
        j++;
    }

    delete[] tempRight;
    delete[] tempLeft;
    return;
}

template <typename T>
void MergeSort(T arr[], int left, int right)
{
    int mid;
    if(left < right)
    {
        mid = left + (right-left)/2;
        MergeSort<T>(arr, left, mid);
        MergeSort<T>(arr, mid+1, right);
        Merge<T>(arr, left, mid, right);
    }

    return;
}


int Find(int tgt)
{
    if(connectivity[tgt] != tgt)
        return connectivity[tgt] = Find(connectivity[tgt]);
    return tgt;
}

void Union(int a, int b)
{
    if(Find(a) == Find(b))
        return;
    else
        connectivity[Find(a)] = connectivity[Find(b)];
    return;
}


int main()
{
    int n, m;
    int from, to, weight;

    cin >> n >> m;
    // initialize disjoint set
    for(int i = 0; i <= n; ++i)
        connectivity[i] = i;
    // initialize edges
    for(int i = 0; i < m; ++i)
    {
        cin >> from >> to >> weight;
        edges[i] = Edge(from, to, weight);
    }
    // sort edges by weight
    MergeSort<Edge>(edges, 0, m-1);

    int treeSize = 0;
    int weightSum = 0;
    int pointer = 0;
    while(treeSize < n-1)
    {
        int currentStart = edges[pointer].from;
        int currentEnd = edges[pointer].to;
        if(Find(currentStart) == Find(currentEnd))
        {
            ++pointer;
            continue;
        }
        else
        {
            Union(currentStart, currentEnd);
            weightSum += edges[pointer].weight;
            ++treeSize;
            ++pointer;
        }
    }
    cout << weightSum;
    
    return 0;
}