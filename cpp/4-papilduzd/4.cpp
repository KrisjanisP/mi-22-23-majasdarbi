#include "utils.hpp"

template <typename T>
T Min(T a, T b)
{
    if (a < b)
        return a;
    else
        return b;
}

int main()
{
    cout << "4. UZDEVUMS" << endl;
    cout << Min("bcd", "abc") << endl;
}