
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
    movq $2, (%rbp)
    mov (%rbp), %r11
    imul $2, %r11
    mov %r11, %r15
    movq %r15, -8(%rbp)
    mov -8(%rbp), %r11
    add (%rbp), %r11
    mov %r11, %rsi
    movq %rsi, -16(%rbp)
    mov -16(%rbp), %rsi
    mov $0, %rax
    lea .int_wformat(%rip), %rdi
    call printf
    leave
    ret
