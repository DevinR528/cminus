
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
    pushq $1092721050
    movsd (%rsp), %xmm5
    movsd %xmm5, -8(%rbp)
    pushq $1065353216
    movsd (%rsp), %xmm5
    movsd %xmm5, (%rsp)
    movsd -8(%rbp), %xmm5
    addss (%rsp), %xmm5
    movsd %xmm5, -16(%rbp)
    cvtss2sd -16(%rbp), %xmm0
    mov $1, %rax
    lea .float_wformat(%rip), %rdi
    call printf
    leave
    ret
