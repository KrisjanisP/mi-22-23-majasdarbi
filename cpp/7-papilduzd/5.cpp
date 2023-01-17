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

int main()
{
    cout << "5. UZDEVUMS" << endl;
    Stack stack;
    if (stack.empty())
        cout << "ir tukšs" << endl;
    else
        cout << "nav tukšs" << endl;
    stack.push(1);
    stack.push(2);
    stack.push(3);
    cout << stack.pop() << endl;
    cout << stack.pop() << endl;
    cout << stack.pop() << endl;
    if (stack.empty())
        cout << "ir tukšs" << endl;
    else
        cout << "nav tukšs" << endl;
}