#include "utils.hpp"

struct Student
{
    string name;
    string surname;
    int age;
    string email;
};

enum class eAction
{
    AddStudent = 1,
    DeleteStudent = 2,
    PrintStudents = 3,
    Exit = 4,
};

eAction promptAction()
{
    std::pair<eAction, std::string> actions[] = {
        {eAction::AddStudent, "Pievienot studentu"},
        {eAction::DeleteStudent, "Nodzēst studentu"},
        {eAction::PrintStudents, "Izdrukāt visus ievadītos studentus"},
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
    cout << "1. UZDEVUMS" << endl;
    vector<Student> students;
    while (true)
    {
        eAction action = promptAction();
        switch (action)
        {
        case eAction::AddStudent:
        {
            Student student;
            input("Ievadiet studenta vārdu: ", student.name);
            input("Ievadiet studenta uzvārdu: ", student.surname);
            input("Ievadiet studenta vecumu: ", student.age);
            input("Ievadiet studenta e-pastu: ", student.email);
            students.push_back(student);
            break;
        }
        case eAction::DeleteStudent:
        {
            string name, surname;
            input("Ievadiet studenta vārdu: ", name);
            input("Ievadiet studenta uzvārdu: ", surname);
            // find student
            int index = -1;
            for (int i = 0; i < students.size(); i++)
            {
                if (students[i].name == name && students[i].surname == surname)
                {
                    index = i;
                    break;
                }
            }
            // remove student
            if (index != -1)
            {
                students.erase(students.begin() + index);
                cout << "Studenta ar šādu vārdu un uzvārdu tika izdzēsts!" << endl;
            }
            else
            {
                cout << "Studenta ar šādu vārdu un uzvārdu neeksistē!" << endl;
            }
            break;
        }
        case eAction::PrintStudents:
        {
            // print students
            for (int i = 0; i < students.size(); i++)
            {
                cout << "Studenta vārds: " << students[i].name << endl;
                cout << "Studenta uzvārds: " << students[i].surname << endl;
                cout << "Studenta vecums: " << students[i].age << endl;
                cout << "Studenta e-pasts: " << students[i].email << endl;
                cout << endl;
            }
            break;
        }
        case eAction::Exit:
        {
            return 0;
        }
        }
    }
}