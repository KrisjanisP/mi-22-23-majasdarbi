#include "utils.hpp"

struct
{
    int hours;
    int minutes;
    int seconds;
} clock_obj;

enum class eAction
{
    SetTime = 1,
    DisplayTime = 2,
    IncrementSecond = 3,
    Exit = 4,
};

eAction promptAction()
{
    std::pair<eAction, std::string> actions[] = {
        {eAction::SetTime, "Uzstādīt laiku"},
        {eAction::DisplayTime, "Attēlot esošo laiku"},
        {eAction::IncrementSecond, "Palielināt laiku pa 1 sekundi"},
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

void correctTime()
{
    clock_obj.minutes += clock_obj.seconds / 60;
    clock_obj.seconds %= 60;
    clock_obj.hours += clock_obj.minutes / 60;
    clock_obj.minutes %= 60;
}

int main()
{
    cout << "4. UZDEVUMS" << endl;
    while (true)
    {
        eAction action = promptAction();
        switch (action)
        {
        case eAction::SetTime:
        {
            input("Ievadiet stundas: ", clock_obj.hours);
            input("Ievadiet minūtes: ", clock_obj.minutes);
            input("Ievadiet sekundes: ", clock_obj.seconds);
            correctTime();
            break;
        }
        case eAction::DisplayTime:
        {
            cout << clock_obj.hours << "h:" << clock_obj.minutes << "m:" << clock_obj.seconds << "s" << endl;
            break;
        }
        case eAction::IncrementSecond:
        {
            clock_obj.seconds++;
            correctTime();
            break;
        }
        case eAction::Exit:
        {
            return 0;
        }
        }
    }
}