#include <iostream>

using namespace std;

class Tuple
{
public:
    long long int a;
    long long int b;
    long long int c;

    Tuple(long long int x = 0, long long int y = 0, long long int z = 0)
    {
        a = x;
        b = y;
        c = z;
    }
};

Tuple ExtendedEuclid(Tuple input)
{
    Tuple result;
    if(input.b == 0)
        return Tuple(1, 0, input.a);
    else
        result = ExtendedEuclid(Tuple(input.b, input.a % input.b, input.c));
    return Tuple(result.b, result.a - input.a / input.b * result.b, result.c);
}

int main()
{
    Tuple equation = Tuple();
    Tuple answer = Tuple();
    while(true)
    {
        // Input.
        cin >> equation.a >> equation.b >> equation.c;

        // Solving equations.
        if(equation.a == 0 && equation.b == 0 && equation.c == 0)
            break;
        answer = ExtendedEuclid(equation);

        // Output.
        if(equation.c % answer.c != 0)
            cout << "No Answer" << endl;
        else
        {      
            // Modifying answers.
            answer.a *= equation.c / answer.c;
            answer.b *= equation.c / answer.c;
            while(answer.a < 0)
            {
                answer.a += equation.b / answer.c;
                answer.b -= equation.a / answer.c;
            }
            while(answer.a - (equation.b / answer.c) >= 0)
            {
                answer.a -= equation.b / answer.c;
                answer.b += equation.a / answer.c;
            }
            cout << answer.a  << ' ' << answer.b << endl;
        }
    }

    return 0;
}