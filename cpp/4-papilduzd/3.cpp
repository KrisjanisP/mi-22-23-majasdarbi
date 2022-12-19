#include "utils.hpp"

void Print(int x)
{
    cout << "int: " << x << endl;
}

void Print(string x)
{
    cout << "string: " << x << endl;
}

void Print(double x)
{
    cout << "double: " << x << endl;
}

void Print(char x)
{
    cout << "char: " << x << endl;
}

int main()
{
    cout << "3. UZDEVUMS" << endl;
    Print(1);
    Print("1");
    Print(1.0);
    Print('1');
}