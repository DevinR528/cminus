
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

.global add
.type add,@function

add:
    pushq %rbp
    mov %rsp, %rbp
    pushq %rdi
    pushq %rsi
    mov -16(%rbp), %rbx
    add -8(%rbp), %rbx
    mov %rbx, %rax
    mov %rax, %rax
    leave
    ret
.global main
.type main,@function

main:
    pushq %rbp
    mov %rsp, %rbp
    pushq $1
    movq $1, %rdi
    movq $1, %rsi
    call add
    movq %rax, -8(%rbp)
    mov -8(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    movq $0, %rax
    leave
    ret
