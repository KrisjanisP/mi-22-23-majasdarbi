#include "utils.hpp"

double sum(double a, double b)
{
    return a + b;
}

double subtract(double a, double b)
{
    return a - b;
}

double multiply(double a, double b)
{
    return a * b;
}

double divide(double a, double b)
{
    return a / b;
}

int main()
{
    cout << "5. UZDEVUMS" << endl;
    int a, b; // first and second number
    input("Ievadiet pirmo skaitli: ", a);
    input("Ievadiet otro skaitli: ", b);
    string op; // operator
    input("Ievadiet operāciju {+,-,*,/}: ", op);
    if (op == "+")
        cout << sum(a, b) << endl;
    else if (op == "-")
        cout << subtract(a, b) << endl;
    else if (op == "*")
        cout << multiply(a, b) << endl;
    else if (op == "/")
        cout << divide(a, b) << endl;
    else
        cout << "Kļūda operācija netika atpazīta!" << endl;
}