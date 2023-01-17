#include "utils.hpp"

struct Point
{
    double x;
    double y;
};

double distance(Point p1, Point p2)
{
    return sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
}

int main()
{
    cout << "2. UZDEVUMS" << endl;
    cout << "Lai iziet no programmas Ctrl+C. lol" << endl;
    while (true)
    {
        // input two points and output their distance
        Point p1, p2;
        input("Ievadiet pirmā punkta x: ", p1.x);
        input("Ievadiet pirmā punkta y: ", p1.y);
        input("Ievadiet otrā punkta x: ", p2.x);
        input("Ievadiet otrā punkta y: ", p2.y);
        cout << "Atstums starp punktiem: " << distance(p1, p2) << endl;
        cout << endl;
    }
}