#include <stdio.h>
#include <stdbool.h>

// Prevent printing an extra message when the user requests program exit.
bool run_again(int *reps)
{
    if (*reps > 0)
    {

        printf("Enter another number to add more repetitions to the series: ");
        scanf("%d", reps);

        return true;
    }
    return false;
}

int main(int argc, char const *argv[])
{
    int reps;
    float add_tally = 1;
    float mix_tally = 1;
    int inc = 0;

    printf("Please input number for repetitions: ");
    scanf("%d", &reps);
    // Don't let the user quit without running once
    reps = (reps < 0) ? reps * -1 : reps;
    do
    {
        for (int i = reps; i > 0; i--)
        {
            inc++;

            add_tally += (1.0 / (float)inc);
            printf("Add series = %f    ", add_tally);

            int sign = inc % 2 == 0 ? -1 : 1;
            mix_tally += ((1.0 / (float)inc) * sign);
            printf("Mix series  = %f\n", mix_tally);
        }
    } while (run_again(&reps));

    return 0;
}
