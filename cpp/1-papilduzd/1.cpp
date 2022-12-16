#include "utils.hpp"

/*
X isn't a prime if it has a 
divisor that isn't 1 or X.
If X does have a divisor, there exists
a divisor that is smaller than the 
square root of X and larger than 1.
*/
bool is_prime(int x) {
    if(x<=2) return false;
    for(int i=2;i*i<=x;i++)
        if(x%i==0)
            return false;
    return true;
}

int main() {
    cout<<"1. UZDEVUMS\n";
    int number;
    input("Ievadiet skaitli: ", number);
    if(is_prime(number)) cout<<number<<" ir pirmskaitlis!\n";
    else cout<<number<<" nav pirmskaitlis!\n";
}