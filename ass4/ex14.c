#include <stdio.h>

void show(int x[], int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
    {
        printf("%5d", x[i]);
    }
    printf("]");
}

int main(int argc, char const *argv[])
{
    int a[8], b[8];

    for (int i = 0; i < 8; i++)
    {
        printf("Enter number for index %d array: ", i);
        scanf("%d", &a[i]);
        if (i == 0)
        {
            b[i] = a[i];
        }
        else
        {
            b[i] = b[i - 1] + a[i];
        }
    }

    int size = sizeof(a) / sizeof(a[0]);
    show(a, size);
    printf("\n");
    show(b, size);

    return 0;
}
