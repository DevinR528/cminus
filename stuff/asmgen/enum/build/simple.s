
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
    sub $12, %rsp
    movq $1, -4(%rbp)
    cmp $0, -4(%rbp)
    cmp $1, -4(%rbp)
    cmp $2, -4(%rbp)
    cmp $69, -4(%rbp)
    leave
    movq $0, %rax
    ret
    leave
    ret
