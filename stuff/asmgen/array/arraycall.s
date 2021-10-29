
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
    pushq %rsi
    movq -8(%rbp), %r11
    movq -16(%rbp), %r9
    imul $8, %r11
    add %r11, %r9
    movq (%r9), %r11
    mov %r11, %rax
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
    pushq $1
    pushq $1
    movq $2, -48(%rbp)
    movq $1, -40(%rbp)
    movq $3, -32(%rbp)
    movq $0, %rdi
    leaq -48(%rbp), %rsi
    call set_idx
    movq %rax, -8(%rbp)
    mov -8(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    movq $1, %rdi
    leaq -48(%rbp), %rsi
    call set_idx
    movq %rax, -16(%rbp)
    mov -16(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    movq $2, %rdi
    leaq -48(%rbp), %rsi
    call set_idx
    movq %rax, -24(%rbp)
    mov -24(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    leave
    ret
