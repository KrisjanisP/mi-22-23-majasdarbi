#include "utils.hpp"
#include <iomanip>

int main() {
    cout<<"3. UZDEVUMS\n";
    srand(time(NULL));
    int a[5][5];
    for(int i=0;i<5;i++)
        for(int j=0;j<5;j++)
            a[i][j] = rand()%100;
    for(int i=0;i<5;i++) {
        for(int j=0;j<5;j++) cout<<setw(4)<<a[i][j];
        cout<<endl;
    }
}