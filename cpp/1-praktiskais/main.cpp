#include <iostream>
#include <vector>
#include "string.h"

#ifdef _WIN32
#include <io.h>
#include "windows.h"
#endif

using namespace kpetrucena;
using std::cin;
using std::cout;
using std::endl;

const string ACTIONS[6] = {
"1. Ievadīt jaunu tekstu",
"2. Noteikt teksta garuma paritāti",
"3. Izvadīt summu no 1 līdz n",
"4. Atrast faktoriāļa n! vērtību",
"5. Izvadīt virkni no otra gala",
"6. Beigt darbību"
};

const string DEF_TXT = "Programmas ir jaraksta cilvekiem, kas tas lasis!";

void clearConsole() {
    #ifdef _WIN32
    system("cls");      // clear console on windows
    #else
    system("clear");    // clear console on linux
    #endif
}

// returns chosen action
int promptAction(string currTxt){
    cout<<"Tagadējais teksts: \n\t\""<<currTxt<<"\""<<endl;
    cout<<"Darbības: "<<endl;
    for(int i=0;i<6;i++)
        cout<<"\t"<<ACTIONS[i]<<endl;
    cout<<"Izvēlieties darbību: ";
    int action;
    cin>>action;
    cin.ignore(1,'\n'); // ignore endline character
    return action;
}

int main() {
    #ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    #endif

    string txt = DEF_TXT;
    while(true) {
        //clearConsole();
        int action = promptAction(txt);

        switch (action) {
            case 1: {
                cout<<"Ievadiet jauno tekstu: ";
                getline(cin,txt);
                break;
            }
            case 2: {
                cout<<"Teksta garums "<<txt.size()<<" ";
                if(txt.size()%2==0) cout<<"ir pāra"<<endl;
                else cout<<"ir nepāra"<<endl;
                system("pause");
                break;
            }
            case 3: {
                int res = 0;
                for(int i=1;i<=txt.size();i++)
                    res += i;
                cout<<"Summa: "<<res<<endl;
                system("pause");
                break;
            }
            case 4: {
                int res = 1;
                for(int i=1;i<=txt.size();i++)
                    res *= i;
                cout<<"Faktoriālis: "<<res<<endl;
                system("pause");
                break;
            }
            case 5: {
                std::string res = txt.toSTDString();
                for(int i=0;i<res.size()/2;i++)
                    std::swap(res[i], res[res.size()-i-1]);
                cout<<res<<endl;
                system("pause");
                break;
            }
            case 6: {
                return 0;
                break;
            }
        }
    }
}