
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
    movq $0, -4(%rbp)
    leaq .bool_false(%rip), %rsi
    cmp $1, -4(%rbp)
    leaq .bool_true(%rip), %rbx
    cmove %rbx, %rsi
    mov $0, %rax
    leaq .str_wformat(%rip), %rdi
    call printf
    leaq .bool_true(%rip), %rsi
    mov $0, %rax
    leaq .str_wformat(%rip), %rdi
    call printf
    leave
    movq $0, %rax
    ret
    leave
    ret
