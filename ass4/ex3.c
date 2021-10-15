#include <stdio.h>

int main(int argc, char const *argv[])
{
    int days;
    printf("Please input number of days: ");
    scanf("%d", &days);

    printf("%d days are %d weeks, %d days.\n", days, days / 7, days % 7);
    return 0;
}
