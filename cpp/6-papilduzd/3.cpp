#include "utils.hpp"

int main()
{
    cout << "3. UZDEVUMS" << endl;

    string *line = new string;
    inputLine("Ievadiet simbolu virkni: ", *line);

    char *point = &((*line)[0]);
    while ((*point) != '\0')
        point++;

    cout << "Simbolu virknes garums: " << point - &((*line)[0]) << endl;
}