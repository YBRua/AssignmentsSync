#include <iostream>
using namespace std;

// subproblem: max matches between i and j;
int match[201][201];

inline bool isMatch(char left, char right)
{
    return (left == '(' && right == ')') || (left == '[' && right == ']');
}

int main()
{
    char input;
    char buffer[201];
    int n = 0;

    while (cin.get(input))
    {
        if (input == '\n')
            break;
        else
        {
            buffer[n] = input;
            ++n;
        }
    }

    for (int i = 0; i < n; ++i)
    {
        match[i][i] = 0;
    }

    for (int size = 1; size < n; ++size)
    {
        for (int i = 0; i < n - size + 1; ++i)
        {
            int j = i + size;
            if (isMatch(buffer[i], buffer[j]))
            {
                match[i][j] = match[i + 1][j - 1] + 2;
            }
            for (int k = i; k < j; ++k)
            {
                // update: match[i][j] = max{match[i][j], match[i][k] + match[k + 1][j]}
                match[i][j] = match[i][j] > (match[i][k] + match[k + 1][j]) ? match[i][j] : (match[i][k] + match[k + 1][j]);
            }
        }
    }
    cout << match[0][n-1];
}