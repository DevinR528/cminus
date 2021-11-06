
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

.global addint
.type addint,@function

addint:
    pushq %rbp
    mov %rsp, %rbp
    pushq %rdi
    pushq %rsi
    mov -16(%rbp), %rdx
    add -8(%rbp), %rdx
    mov %rdx, %r8
    mov %r8, %rax
    leave
    ret
.global fooint
.type fooint,@function

fooint:
    pushq %rbp
    mov %rsp, %rbp
    pushq %rdi
    pushq %rsi
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call addint
    mov %rax, %rax
    leave
    ret
.global main
.type main,@function

main:
    pushq %rbp
    mov %rsp, %rbp
    pushq $1
    pushq $1
    pushq $1
    movq $10, -8(%rbp)
    movq $111, -16(%rbp)
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call fooint
    movq %rax, -24(%rbp)
    mov -24(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    leave
    ret
