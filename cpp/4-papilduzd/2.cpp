#include "utils.hpp"

int sum(int n)
{
    if (n <= 0)
        return 0;
    return n + sum(n - 1);
}

int main()
{
    cout << "2. UZDEVUMS" << endl;
    int n;
    input("Ievadiet skaitli n: ", n);
    cout << "Summa 1..n: " << sum(n) << endl;
}