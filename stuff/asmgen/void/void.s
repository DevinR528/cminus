
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
    push %rbp
    mov %rsp, %rbp
    sub $8, %rsp
    sub $8, %rsp
    sub $8, %rsp

    mov $2, %r14
    mov %r14, -0(%rbp)
    mov $2, %r14            # remove
    mov -0(%rbp), %r11
    imul %r14, %r11

    mov %r11, -8(%rbp)      # add save result

    mov %r11, %r10
    mov -8(%rbp), %r10
    mov -8(%rbp), %r11      # remove
    add -0(%rbp), %r11

    mov %r11, -16(%rbp)     # add save result

    mov -16(%rbp), %r10     # remove

    mov -16(%rbp), %rsi
    mov $0, %rax
    lea .int_wformat(%rip), %rdi
    call printf
    leave
    ret
