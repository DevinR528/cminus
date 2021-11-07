
.text

.char_wformat: .string "%c\n"
.int_wformat: .string "%d\n"
.float_wformat: .string "%f\n"
.str_wformat: .string "%s\n"
.char_rformat: .string "%c"
.int_rformat: .string "%d"
.float_rformat: .string "%f"
.bool_true: .string "true"
.bool_false: .string "false"
.bool_test: .quad 1

.global main
.type main,@function

main:
    pushq %rbp
    mov %rsp, %rbp
    pushq $1
    pushq $1
    pushq $1
    pushq $1
    movq $3, -32(%rbp)
    movq $10, -8(%rbp)
    movq $5, -24(%rbp)
    mov -24(%rbp), %rbx
    imul -8(%rbp), %rbx
    mov %rbx, %r9
    sub $3, %r9
    mov %r9, %rbx
    movq %rbx, -16(%rbp)
    mov -16(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    mov -24(%rbp), %rbx
    imul -8(%rbp), %rbx
    mov %rbx, %r9
    mov -16(%rbp), %rbx
    sub %rbx, %r9
    mov %rbx, %rsi
    mov %rsi, %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    movq $0, %rax
    leave
    ret
