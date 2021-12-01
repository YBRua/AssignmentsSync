/**
  *  ACKNOWLEDGEMENT:
  *  The idea of this algorithm is originally proposed by my roommate Eliphat.
  */
#include <iostream>

#define SIZE 1000000

using namespace std;

int CakeTeaSet[SIZE]; // Disjoint Set

void Init()
{
    for(int i = 0; i < SIZE; ++i)
    {
        CakeTeaSet[i] = i;
    }
}

int Find(int obj)
{
    if(CakeTeaSet[obj] == obj)
        return CakeTeaSet[obj];
    else
        return CakeTeaSet[obj] = Find(CakeTeaSet[obj]);
}

void Union(int r1, int r2)
{
    if(Find(r1) == Find(r2))
        return;
    else
        CakeTeaSet[r1] = CakeTeaSet[r2];
}

int main()
{
    int m;
    int n;

    int start;
    int end;

    cin >> n >> m;

    Init();

    for(int i = 0; i < m; ++i)
    {
        cin >> start >> end;
        if(Find(start) == Find(end))
        {
            cout << "Yes";
            return 0;
        }
        else
        {
            Union(start, end);
        }
    }
    cout << "No";

    return 0;
}