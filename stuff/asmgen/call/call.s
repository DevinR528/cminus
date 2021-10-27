
.text

.char_wformat: .string "%c\n"
.int_wformat: .string "%d\n"
.float_wformat: .string "%f\n"
.str_wformat: .string "%s\n"
.char_rformat: .string "%c"
.int_rformat: .string "%d"
.float_rformat: .string "%f"

.global add
.type add,@function

add:
    pushq %rbp
    mov %rsp, %rbp
    pushq %rdi
    pushq %rsi
    mov -16(%rbp), %r9
    add -8(%rbp), %r9
    mov %r9, %r11
    mov %r11, %rax
    leave
    ret
.global main
.type main,@function

main:
    pushq %rbp
    mov %rsp, %rbp
    movq $1, %rdi
    movq $1, %rsi
    call add
    movq %rax, -24(%rbp)
    mov -24(%rbp), %rsi
    mov $0, %rax
    lea .int_wformat(%rip), %rdi
    call printf
    leave
    ret
