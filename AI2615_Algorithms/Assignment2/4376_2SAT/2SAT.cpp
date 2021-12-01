/**
 * Graph-based solution to 2SAT:
 * a or b == (not a \to b) and (not b \to a)
 * a and (not a) cannot be in the same strongly connected component
 */
#define SIZE 2000000

#include <iostream>
#include <vector>

using namespace std;

vector<int> adjList[SIZE];
vector<int> adjListReversed[SIZE];

int group[SIZE]; // which strongly connected component x is in
bool visited[SIZE];
vector<int> label; // post-order label in round 1 DFS

inline int IndexOf(int x)
{
    return 2 * (x-1);
}
inline int IndexOfNot(int x)
{
    return 2*(x-1) + 1;
}

void ExploreAndLabel(int node)
{
    // basic dfs
    visited[node] = true;
    for(int i = 0; i < adjList[node].size(); ++i)
    {
        if(!visited[adjList[node][i]])
            ExploreAndLabel(adjList[node][i]);
    }
    // post-order labeling
    label.push_back(node);

    return;
}

void DFSRound1()
{
    for(int i = 0; i < SIZE; ++i)
        visited[i] = false;
    for(int i = 0; i < SIZE; ++i)
        if(!visited[i])
            ExploreAndLabel(i);
    return;
}

void FindStronglyConnectedComponents(int node, int grp)
{
    group[node] = grp;
    for(int i = 0; i < adjListReversed[node].size(); ++i)
    {
        if(group[adjListReversed[node][i]] == -1)
            FindStronglyConnectedComponents(adjListReversed[node][i], grp);
    }
}

void DFSRound2()
{
    for(int i = 0; i < SIZE; ++i)
        group[i] = -1;
    for(int i = 0, j = 0; i < SIZE; ++i)
    {
        int node = label[SIZE - i - 1]; // find the node with highest order
        if(group[node] == -1)
            FindStronglyConnectedComponents(node, j++);
    }
}

void KosarajuAlgo()
{
    DFSRound1();
    DFSRound2();

    return;
}

inline int FastInput()
{
    int value = 0;
    char temp;
    temp = getchar();
    while(temp < '0' || temp > '9')
        temp = getchar();
    while(temp >= '0' && temp <= '9')
    {
        value *= 10;
        value += temp - '0';
        temp = getchar();
    }

    return value;
}

int main()
{
    int numVars;
    int numConstrs;

    int operand_1;
    int value_1;
    int operand_2;
    int value_2;
    int start;
    int end;


    scanf("%d", &numVars);
    scanf("%d", &numConstrs);

    // take input and construct implication graph
    // x_k uses index 2k; (not x_k) uses index 2k+1
    for(int i = 0; i < numConstrs; ++i)
    {
        operand_1 = FastInput();
        value_1 = FastInput();
        operand_2 = FastInput();
        value_2 = FastInput();

        start = value_1 == 1 ? IndexOfNot(operand_1) : IndexOf(operand_1);
        end = value_2 == 1 ? IndexOf(operand_2) : IndexOfNot(operand_2);
        adjList[start].push_back(end);
        adjListReversed[end].push_back(start);

        start = value_2 == 1 ? IndexOfNot(operand_2) : IndexOf(operand_2);
        end = value_1 == 1 ? IndexOf(operand_1) : IndexOfNot(operand_1);
        adjList[start].push_back(end);
        adjListReversed[end].push_back(start);
    }

    KosarajuAlgo();

    for(int i = 0; i < SIZE; i+=2)
    {
        if(group[i] == group[i+1])
        {
            cout << "No";
            return 0;
        }
    }
    
    cout << "Yes";
    return 0;
}