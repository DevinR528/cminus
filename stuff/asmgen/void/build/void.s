
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
    movq $2, -8(%rbp)
    mov -8(%rbp), %rcx
    imul $2, %rcx
    mov %rcx, %r12
    movq %r12, -16(%rbp)
    mov -16(%rbp), %rcx
    add -8(%rbp), %rcx
    mov %rcx, %r12
    movq %r12, -24(%rbp)
    mov -24(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    leave
    movq $0, %rax
    ret
    leave
    ret
