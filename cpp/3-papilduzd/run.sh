TASK=3
read -p "Izvēlieties uzdevumu [1-5]: " TASK
FILENAME=${TASK}.cpp
g++ --std=c++11 ${FILENAME}
./a.out
rm -f ./a.out