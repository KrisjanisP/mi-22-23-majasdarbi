#include "utils.hpp"

int main() {
    cout<<"3. UZDEVUMS\n";
    int n;
    input("Lūdzu, ievadiet n: ", n);
    // acīmredzamas izteiksmes:
    int sum_three = ((n/3)*((n/3)+1)/2)*3;
    int sum_five = ((n/5)*((n/5)+1)/2)*5;
    int sum_fifteen = ((n/15)*((n/15)+1)/2)*15;
    cout<<"Summa: "<<sum_three+sum_five-sum_fifteen<<endl;
}