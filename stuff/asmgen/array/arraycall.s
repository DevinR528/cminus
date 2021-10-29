
.text

.char_wformat: .string "%c\n"
.int_wformat: .string "%d\n"
.float_wformat: .string "%f\n"
.str_wformat: .string "%s\n"
.char_rformat: .string "%c"
.int_rformat: .string "%d"
.float_rformat: .string "%f"

.global set_idx
.type set_idx,@function

set_idx:
    pushq %rbp
    mov %rsp, %rbp
    pushq %rdi
    pushbig %rsi
    mov -24(%rbp), %rax
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
    pushq $1
    movq $2, -64(%rbp)
    movq $1, -56(%rbp)
    movq $3, -48(%rbp)
    movq $1, %rdi
    movbig -64(%rbp), %rsi
    call set_idx
    movq %rax, -40(%rbp)
    mov -40(%rbp), %rsi
    mov $0, %rax
    lea .int_wformat(%rip), %rdi
    call printf
    leave
    ret
