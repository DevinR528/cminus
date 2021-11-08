
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
    pushq $1092721050
    movsd (%rsp), %xmm3
    movsd %xmm3, -8(%rbp)
    pushq $1065353216
    movsd (%rsp), %xmm3
    movsd %xmm3, (%rsp)
    movsd -8(%rbp), %xmm3
    addss (%rsp), %xmm3
    movsd %xmm3, -16(%rbp)
    cvtss2sd -16(%rbp), %xmm0
    mov $1, %rax
    leaq .float_wformat(%rip), %rdi
    call printf
    leave
    movq $0, %rax
    ret
    leave
    ret
