#include "utils.hpp"

struct Node
{
    int num;
    Node *next;
};

struct Stack
{
    Node *head = NULL;

    void push(int num)
    {
        Node *node = new Node;
        node->num = num;
        node->next = head;
        head = node;
    }

    int pop()
    {
        int num = head->num;
        Node *node = head->next;
        delete head;
        head = node;
        return num;
    }

    bool empty()
    {
        return head == NULL;
    }
};

enum class eAction
{
    AddElement = 1,
    PopElement = 2,
    Empty = 3,
    Exit = 4,
};

eAction promptAction()
{
    std::pair<eAction, std::string> actions[] = {
        {eAction::AddElement, "Ievietot jaunu elementu"},
        {eAction::PopElement, "Izņemt elementu no steka"},
        {eAction::Empty, "Pārbaudīt, vai steks ir tukšs"},
        {eAction::Exit, "Beigt Programmu"}};

    cout << "Izvēlieties darbību:\n";
    for (auto action : actions)
    {
        cout << "\t" << (int)action.first << ". " << action.second << endl;
    }

    eAction result;
    input("Izvēlieties darbību [1-4]: ", (int &)result, [](int x) -> bool
          { return x >= 1 && x <= 4; });
    return result;
}

int main()
{
    cout << "5. UZDEVUMS" << endl;
    Stack stack;
    while(true) {
        eAction action = promptAction();
        // switch and use stack depending on action
        switch (action)
        {
        case eAction::AddElement:
        {
            int num;
            input("Ievadiet skaitli: ", num);
            stack.push(num);
            break;
        }
        case eAction::PopElement:
        {
            cout << "Izņemts elements: " << stack.pop() << endl;
            break;
        }
        case eAction::Empty:
        {
            if(stack.empty())
                cout << "Steks ir tukšs." << endl;
            else
                cout << "Steks nav tukšs." << endl;
            break;
        }
        case eAction::Exit:
            return 0;
        }
    }
}