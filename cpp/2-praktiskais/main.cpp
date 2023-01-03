#include "utils.hpp"

using namespace std;

enum action
{
    number_prompt_action = 1,
    queue_action,
    stack_action,
    double_stack_action,
    list_action,
    output_action,
    stop_action
};

action prompt_action(const vector<int> &curr_numbers)
{
    const string actions[] = {
        "1: Ievadīt skaitļus",
        "2: Rinda",
        "3: Steks",
        "4: Divi steki (Rinda)",
        "5: Saraksts",
        "6: Izvadīt skaitļu virkni",
        "7: Beigt darbību"};
    cout << "Tagadējie skaitļi: \n\t" << to_string(curr_numbers) << "" << endl;
    cout << "Darbības: " << endl;
    for (int i = 0; i < 7; i++)
        cout << "\t" << actions[i] << endl;
    int result;
    input("Izvēlieties darbību [1-7]: ", result, [](int x) -> bool
          { return x >= 1 && x <= 7; });
    return action(result);
}

void clear_console()
{
    #ifdef _WIN32
    system("cls");      // clear console on windows
    #else
    system("clear");    // clear console on linux
    #endif
}

void pause_console()
{
    cout << "Nospiediet enter, lai turpināt!" << endl;
    cin.ignore(69420, '\n'); // clear input buffer
}

vector<int> prompt_numbers()
{
    cout << "Ievadiet skaitļus: ";
    string line;
    getline(cin, line);
    stringstream ss(line);
    vector<int> result;
    int x;
    while (ss >> x)
    {
        result.push_back(x);
    }
    return result;
}

void do_queue_action(const vector<int> &curr_numbers)
{
    queue<int> rinda;
    for (int x : curr_numbers)
        rinda.push(x);
    vector<int> result;
    while (!rinda.empty())
    {
        result.push_back(rinda.front());
        rinda.pop();
    }
    cout << "Ņemot ārā no rindas: " << to_string(result) << endl;
}

void do_stack_action(const vector<int> &curr_numbers)
{
    stack<int> steks;
    for (int x : curr_numbers)
        steks.push(x);
    vector<int> result;
    while (!steks.empty())
    {
        result.push_back(steks.top());
        steks.pop();
    }
    cout << "Ņemot ārā no steka: " << to_string(result) << endl;
}

void do_double_stack_action(const vector<int> &curr_numbers)
{
    stack<int> steki[2];
    for (int x : curr_numbers)
        steki[0].push(x);
    while (!steki[0].empty())
    {
        steki[1].push(steki[0].top());
        steki[0].pop();
    }
    vector<int> result;
    while (!steki[1].empty())
    {
        result.push_back(steki[1].top());
        steki[1].pop();
    }
    cout << "Liekot iekšā, ņemot ārā, liekot iekšā, ņemot ārā no steka: " << to_string(result) << endl;
}

void do_list_action(const vector<int> &curr_numbers)
{
    list<int> saraksts;
    for (int x : curr_numbers)
        saraksts.push_back(x);
    vector<int> result;
    while (!saraksts.empty())
    {
        result.push_back(saraksts.back());
        saraksts.pop_back();
    }
    cout << "Ņemot ārā no saraksta aizmugures: " << to_string(result) << endl;
    for (int x : curr_numbers)
        saraksts.push_back(x);
    result.clear();
    while (!saraksts.empty())
    {
        result.push_back(saraksts.front());
        saraksts.pop_front();
    }
    cout << "Ņemot ārā no saraksta priekšas: " << to_string(result) << endl;
}

void do_output_action(const vector<int> &curr_numbers)
{
    cout << "Vektorā uzglabātie skaitļi: " << to_string(curr_numbers) << endl;
}

int main()
{
    vector<int> curr_numbers = {4, 5, 6, 7, 8};
    while (true)
    {
        clear_console();
        action action = prompt_action(curr_numbers);
        switch (action)
        {
        case number_prompt_action:
        {
            curr_numbers = prompt_numbers();
            break;
        }
        case queue_action:
        {
            do_queue_action(curr_numbers);
            pause_console();
            break;
        }
        case stack_action:
        {
            do_stack_action(curr_numbers);
            pause_console();
            break;
        }
        case double_stack_action:
        {
            do_double_stack_action(curr_numbers);
            pause_console();
            break;
        }
        case list_action:
        {
            do_list_action(curr_numbers);
            pause_console();
            break;
        }
        case output_action:
        {
            do_output_action(curr_numbers);
            pause_console();
            break;
        }
        case stop_action:
            return 0;
        }
    }
}