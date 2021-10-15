#include <stdbool.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{
    int mod_by;
    int to_mod = 1;
    printf("Please enter number for moduli: ");
    scanf("%d", &mod_by);
    printf("Please enter number to be first operand: ");
    scanf("%d", &to_mod);

    do
    {
        printf("%d %% %d = %d\n", to_mod, mod_by, to_mod % mod_by);
        printf("Enter another number or 0 or less: ");
        scanf("%d", &to_mod);
    } while (to_mod > 0);

    return 0;
}
