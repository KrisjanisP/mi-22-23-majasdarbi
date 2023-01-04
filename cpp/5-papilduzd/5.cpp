#include "utils.hpp"

class DateTime {
public:
    int year;
    int month;
    int day;
    int hour;
    int minute;
    DateTime(){}
    DateTime(int year, int month, int day, int hour, int minute):
        year(year),month(month),day(day),hour(hour),minute(minute){}
    string toString() const;
    void readFrom(istream& is) {
        is.read(reinterpret_cast<char*>(this),sizeof(*this));
    }
    void saveTo(ofstream& os) {
        os.write(reinterpret_cast<char*>(this),sizeof(*this));
    }
};

istream& operator>>(istream& is, DateTime& rhs) {
    is>>rhs.day;
    is.ignore(1,'.');
    is>>rhs.month;
    is.ignore(1,'.');
    is>>rhs.year;

    is>>rhs.hour;
    is.ignore(1,':');
    is>>rhs.minute;

    return is;
}

ostream& operator<<(ostream& os, const DateTime& rhs) {
    os<<rhs.day<<"."<<rhs.month<<"."<<rhs.year<<" "<<rhs.hour<<":"<<rhs.minute;
    return os;
}

string DateTime::toString() const{
    stringstream ss;
    ss<<*this;
    return ss.str();
}

enum eAction {
    a_input = 1,
    a_save,
    a_read,
    a_exit
};

void saveDateTime(const DateTime& dt) {
    ofstream file("datetime.txt", ios::binary);
    if(file.is_open()){
        string dateTimeStr = dt.toString();
        file.write(dateTimeStr.data(), dateTimeStr.size());
        file.close();
        cout<<"Datums "<<dt<<" saglabāts \"datetime.txt\".\n";
    }else {
        cout<<"Neizdevās atvērt \"datetime.txt\".\n";
    }
}

eAction promptAction() {
    pair<eAction, string> actions[] = {
        {a_input,"Ievadit datumu un laiku"},
        {a_save,"Saglabāt binārajā failā"},
        {a_read,"Nolasīt un izdrukāt uz ekrāna"},
        {a_exit,"Beigt programmu"}
    };

    cout<<"Izvēlieties darbību:\n";
    for(auto action: actions) {
        cout<<"\t"<<action.first<<". "<<action.second<<endl;
    }

    eAction result;
    input("Izvēlieties darbību [1-4]: ", (int&)result, [](int x)->bool
        {return x>=1&&x<=4;});
    return result;
}

DateTime promptDateTime() {
    DateTime dt;
    input("Ievadiet datumu [dd.mm.yyyy hh:mm]: ", dt);
    return dt;
}

int main() {
    cout<<"5. UZDEVUMS"<<endl;
    DateTime current(2023,01,04,23,19);
    while(true) {
        clearConsole();
        cout<<"datums: "<<current<<endl;
        eAction action = promptAction();
        switch(action) {
            case a_input: {
                current = promptDateTime();
                break;
            }
            case a_save: {
                ofstream file("datetime.bin", ios::binary);
                assert(file.is_open());
                current.saveTo(file);
                file.close();
                break;
            }
            case a_read: {
                ifstream file("datetime.bin", ios::binary);
                assert(file.is_open());
                current.readFrom(file);
                file.close();
                break;
            }
            case a_exit: {
                return 0;
            }
        }
    }
}