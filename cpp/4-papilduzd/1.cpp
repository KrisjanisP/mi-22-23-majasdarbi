#include "utils.hpp"

int main()
{
    cout << "1. UZDEVUMS" << endl;
    string line;
    inputLine("Ievadiet simbolu virkni: ", line);
    for (int i = 0; i < line.size() / 2; i++)
        swap(line[i], line[line.size() - 1 - i]);
    cout << line << endl;
}