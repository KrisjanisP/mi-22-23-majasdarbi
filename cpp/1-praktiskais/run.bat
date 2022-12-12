g++ string.cpp main.cpp bigint.cpp -o ./bin/exe
if %errorlevel% neq 0 exit /b %errorlevel%
.\bin\exe.exe