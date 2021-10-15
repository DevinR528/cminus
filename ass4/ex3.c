#include <stdio.h>

int main(int argc, char const *argv[])
{
    int days, weeks, days_left;
    printf("Please input number of days: ");
    scanf("%d", &days);

    // Don't do 2 somewhat expensive math ops when you can do one
    asm(
        "mov %2, %%eax;"
        "cdq;"
        "mov $7, %%ebx;"
        "idiv %%ebx"
        : "=a"(weeks), "=d"(days_left)
        : "g"(days));

    printf("%d days is %d weeks, %d days.\n", days, weeks, days_left);
    return 0;
}
