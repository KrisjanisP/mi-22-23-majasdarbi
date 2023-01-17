#include "utils.hpp"

int main()
{
    cout << "1. UZDEVUMS" << endl;
    int *a = new int, *b = new int;
    input("Ievadiet pirmo skaitli: ", *a);
    input("Ievadiet otro skaitli: ", *b);
    cout << "Summa: " << *a + *b << endl;
}