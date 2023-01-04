#include "utils.hpp"

int main() {
    cout<<"3. UZDEVUMS"<<endl;
    ifstream file("file4.txt");
    if(file.is_open()) {
        int res = 0;
        char c;
        while(file>>c) {
            if((c>='a'&&c<='z')||(c>='A'&&c<='Z'))
                res++;
        }
        cout<<"Burtu skaits: "<<res<<endl;
        file.close();
    } else {
        cout<<"Neizdevās atvērt \"file4.txt\".\n";
    }
}