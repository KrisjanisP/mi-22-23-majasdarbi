#include "utils.hpp"

int main() {
    cout<<"1. UZDEVUMS"<<endl;
    string text;
    inputLine("Ievadiet tekstu: ", text);
    ofstream output("text.txt");
    if(output.is_open()) {
        output<<text;
        output.close();
        cout<<"Tika izveidots fails \"text.txt\".\n";
    } else {
        cout<<"Failu neizdevās atvērt.\n";
    }
}