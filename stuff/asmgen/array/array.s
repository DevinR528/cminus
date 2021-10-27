
.text

.char_wformat: .string "%c\n"
.int_wformat: .string "%d\n"
.float_wformat: .string "%f\n"
.str_wformat: .string "%s\n"
.char_rformat: .string "%c"
.int_rformat: .string "%d"
.float_rformat: .string "%f"

.global main
.type main,@function

main:
    pushq %rbp
    mov %rsp, %rbp

    movq $0, -32(%rbp)
    movq $9, -24(%rbp)
    movq $11, -16(%rbp)
    movq $15, -8(%rbp)

    mov -32(%rbp), %rsi
    mov $0, %rax
    lea .int_wformat(%rip), %rdi
    call printf

    mov -24(%rbp), %rsi
    mov $0, %rax
    lea .int_wformat(%rip), %rdi
    call printf

    mov -16(%rbp), %rsi
    mov $0, %rax
    lea .int_wformat(%rip), %rdi
    call printf

    mov -8(%rbp), %rsi
    mov $0, %rax
    lea .int_wformat(%rip), %rdi
    call printf

    leave
    ret
