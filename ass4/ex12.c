#include <stdio.h>

int main(int argc, char const *argv[])
{
    int reps;
    printf("Please input number for repetitions: ");
    scanf("%d", &reps);

    float add_tally = 1;
    float mix_tally = 1;
    int inc = 0;
    while (reps != inc)
    {
        inc++;
        add_tally += (1.0 / (float)inc);
        printf("Add series = %f    ", add_tally);

        int sign = inc % 2 == 0 ? -1 : 1;
        mix_tally += ((1.0 / (float)inc) * sign);
        printf("Mix series  = %f\n", mix_tally);
    }

    return 0;
}
