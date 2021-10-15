#include <stdio.h>

const int DUNBAR = 150;
const int RL_FRIEDNDS = 5;

int main(int argc, char const *argv[])
{
    int week = 1;
    int friends = RL_FRIEDNDS;
    while (friends < DUNBAR)
    {
        friends = (friends - week) * 2;
        printf("Week %d Professor Rabnud has %d friends\n", week, friends);
        week++;
    }
    return 0;
}
