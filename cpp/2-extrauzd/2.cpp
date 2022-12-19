#include "utils.hpp"

int main() {
    cout<<"2.1. - 2.3. UZDEVUMS\n";

    cout<<"2.1. UZDEVUMS\n";
    int width, height;
    input("Ievadiet taisnstūra platumu: ", width);
    input("Ievadiet taisnstūra augstumu: ", height);
    for(int i=0;i<height;i++) {
        for(int j=0;j<width;j++) {
            if(!i||!j||i==height-1||j==width-1) cout<<'*';
            else cout<<' ';
        }
        cout<<'\n';
    }

    cout<<"2.2. UZDEVUMS\n";
    int x, y;
    input("Ievadiet krustpunkta x koordinati: ", x);
    input("Ievadiet krustpunkta y koordinati: ", y);
    cout<<"Taisnstūra augšējais kreisais stūris: ("<<x-width/2.0<<","<<y+height/2.0<<")"<<endl;
    cout<<"Taisnstūra augšējais labais stūris: ("<<x+width/2.0<<","<<y+height/2.0<<")"<<endl;
    cout<<"Taisnstūra apakšējais labais stūris: ("<<x+width/2.0<<","<<y-height/2.0<<")"<<endl;
    cout<<"Taisnstūra apakšējais kreisais stūris: ("<<x-width/2.0<<","<<y-height/2.0<<")"<<endl;

    cout<<"2.3. UZDEVUMS\n";
    int a_x, a_y;
    input("Ievadiet punkta A x koordinati: ", a_x);
    input("Ievadiet punkta A y koordinati: ", a_y);
    if(a_x>=x-width/2.0&&a_x<=x+width/2.0&&a_y>=y-height/2.0&&a_y<=y+height/2.0)
        cout<<"šis punkts ietilpst iekš dotā taisnstūra\n";
    else
        cout<<"šis punkts NEietilpst iekš dotā taisnstūra\n";

}