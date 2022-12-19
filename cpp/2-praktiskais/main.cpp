#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

const string ACTIONS[9] = {
"1: Ievadīt skaitļus",
"2: Rinda",
"3: Steks",
"4: Divi steki (Rinda)",
"5: Saraksts",
"6: Izvadīt skaitļu virkni",
"7: Beigt darbību"
};

template<typename T>
string to_string(vector<T> vec) {
    stringstream ss;
    for(int i=0;i<vec.size();i++){
        ss<<vec[i];
        if(i<vec.size()-1) ss<<",";
    }
    return "{"+ss.str()+"}";
}


template<typename T, typename Lambda>
void input(const string& prompt, T &variable, Lambda correct) {
    cout<<prompt;
    cin>>variable;
    if(cin.fail()||!correct(variable)) {
        cin.clear();
        cin.ignore(69420, '\n');
        std::cout<<"Ievades kļūda!\n";
        return input(prompt,variable,correct);
    }
    cin.ignore(69420, '\n'); // clear buffer
}

int prompt_action(const vector<int>& curr_numbers) {
    cout<<"Tagadējie skaitļi: \n\t"<<to_string(curr_numbers)<<""<<endl;
    cout<<"Darbības: "<<endl;
    for(int i=0;i<7;i++)
        cout<<"\t"<<ACTIONS[i]<<endl;
    int action;
    input("Izvēlieties darbību [1-7]: ",action,[](int action)->bool {return action>=1&&action<=7;});
    return action;
}

vector<int> prompt_numbers() {
    cout<<"Ievadiet skaitļus: ";
    string line;

    getline(cin, line);
    stringstream ss(line);
    vector<int> result;
    int x;
    while(ss>>x){result.push_back(x);}
    return result;
}

void clear_console() {
    system("clear");
}

int main () {
    vector<int> curr_numbers = {4,5,6,7,8};
    while(true) {
        clear_console();
        int action = prompt_action(curr_numbers);
        switch(action) {
            case 1:
                curr_numbers = prompt_numbers();
                break;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
                return 0;
        }
    }
}