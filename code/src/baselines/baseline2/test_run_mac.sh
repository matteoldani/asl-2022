output=$(gcc-11 -I/opt/homebrew/opt/openblas/include -pthread -O3 -Wall test_environment.c -o test_environment.out -L/opt/homebrew/opt/openblas/lib  -lm -lpthread -lopenblas)
if [[ $? != 0 ]]; then
    echo -e "Error:\n$output"
else
    ./test_environment.out
fi
rm test_environment.out