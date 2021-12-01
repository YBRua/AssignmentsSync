#include <iostream>

using namespace std;

char a[4000];
int length_a = 0;
char b[4000];
int length_b = 0;
int dist[4000][4000];

inline int minOf3(int a, int b, int c)
{
    int min;
    min = a < b ? a : b;
    min = min < c ? min : c;

    return min;
}

inline int diff(char a, char b)
{
    return a == b ? 0 : 1;
}

int main()
{
    int x; // cost of delete or insert a character
    int y; // cost of substituting a character
    char temp;

    cin >> x >> y;
    cin >> a;
    cin >> b;
    for (int i = 0; a[i] != '\0'; ++i)
        length_a += 1;
    for (int i = 0; b[i] != '\0'; ++i)
        length_b += 1;

    for (int i = 0; i <= length_a; ++i)
        dist[i][0] = i * x;
    for (int j = 0; j <= length_b; ++j)
        dist[0][j] = j * x;

    for (int i = 1; i <= length_a; ++i)
    {
        for (int j = 1; j <= length_b; ++j)
            dist[i][j] = minOf3(dist[i-1][j] + x, dist[i][j-1] + x, dist[i-1][j-1] + y * diff(a[i-1], b[j-1]));
    }

    cout << dist[length_a][length_b];

    return 0;
}