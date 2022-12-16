#include "utils.hpp"

int main() {
    cout<<"2. UZDEVUMS\n";
    int width, height;
    input("Input rectangle's width: ", width);
    input("Input rectangle's height: ", height);
    cout<<"Rectangle's area: "<<width*height;
    cout<<"Rectangle's perimeter: "<<2*(width+height);
}