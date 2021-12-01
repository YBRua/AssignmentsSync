#include <iostream>
#include <vector>
#define SIZE 205
#define PARENT 0
#define LEFT 1
#define RIGHT 2

using namespace std;

int tree[SIZE][3];
int nodeWeight[SIZE];
int result[SIZE][SIZE];

void search(int root, int m)
{
    if (tree[root][LEFT] == 0 && tree[root][RIGHT] == 0)
    {
        // leaf
        for (int i = 1; i <= m; ++i)
        {
            result[root][i] = nodeWeight[root];
        }
    }
    else
    {
        // non-leaf
        if (tree[root][LEFT] != 0)
            search(tree[root][LEFT], m);
        if (tree[root][RIGHT] != 0)
            search(tree[root][RIGHT], m);
        int pointer = 0;
        if (tree[root][LEFT] == 0 && tree[root][RIGHT] != 0)
            // has only one child on the right
            pointer = RIGHT;
        else if (tree[root][LEFT] != 0 && tree[root][RIGHT] == 0)
            // has only one child on the left
            pointer = LEFT;
        if (pointer)
        {
            // has only one child
            result[root][1] = nodeWeight[root];
            for (int i = 2; i <= m; ++i)
            {
                result[root][i] = result[root][i] > result[tree[root][pointer]][i - 1] 
                    ? result[root][i] 
                    : result[tree[root][pointer]][i - 1] + nodeWeight[root];
            }
        }
        else
        {
            // has two children
            result[root][1] = nodeWeight[root];
            for (int i = 2; i <= m; ++i)
            {
                for (int k = 0; k <= i - 1; ++k)
                {
                    int left = result[tree[root][LEFT]][k];
                    int right = result[tree[root][RIGHT]][i-1-k];
                    result[root][i] = result[root][i] > nodeWeight[root] + left + right
                        ? result[root][i]
                        : nodeWeight[root] + left + right;
                }
            }
        }
    }
}

int main()
{
    int n; // total num of nodes
    int m; // num of nodes to retain

    nodeWeight[1] = 0;
    tree[1][PARENT] = 1;

    cin >> n >> m;
    for (int i = 2; i <= n; ++i)
        cin >> nodeWeight[i];
    for (int i = 1; i < n; ++i)
    {
        int begin;
        int end;
        cin >> begin >> end;
        if (tree[begin][PARENT] != 0)
        {
            if (tree[begin][LEFT] == 0)
                tree[begin][LEFT] = end;
            else
                tree[begin][RIGHT] = end;
            tree[end][PARENT] = begin;
        }
        else
        {
            if (tree[end][LEFT] == 0)
                tree[end][LEFT] = begin;
            else
                tree[end][RIGHT] = begin;
            tree[begin][PARENT] = end;
        }
    }

    search(1, m+1);

    cout << result[1][m+1];

    return 0;
}