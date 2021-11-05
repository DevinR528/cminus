
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

.global main
.type main,@function

main:
    pushq %rbp
    mov %rsp, %rbp
    push $1
    movl $0, -4(%rbp)
    leaq .bool_false(%rip), %rsi
    cmp $1, -4(%rbp)
    leaq .bool_true(%rip), %rax
    cmove %rax, %rsi
    mov $0, %rax
    leaq .str_wformat(%rip), %rdi
    call printf
    leave
    ret
