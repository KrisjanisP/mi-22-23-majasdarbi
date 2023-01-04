#include "utils.hpp"

int main() {
    cout<<"4. UZDEVUMS"<<endl;
    ifstream first_task("1.cpp");
    if(first_task.is_open()) {
        string original;
        getline(first_task, original, '\0');

        char remove;
        input("Ievadiet burtu, ko izņemt: ", remove);
        
        string modified;
        for(int i=0;i<original.size();i++) {
            if(original[i]==remove) continue;
            modified += original[i];
        }

        ofstream output("file5.txt");
        if(output.is_open()) {
            output<<modified;
            output.close();
            cout<<"Tika izveidots fails \"file5.txt\".\n";
        } else {
            cout<<"Failu \"file5.txt\" neizdevās atvērt.\n";
        }

        first_task.close();
    } else {
        cout<<"Neizdevās atvērt \"1.cpp\".\n";
    }
}