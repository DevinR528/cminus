
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
.smaller: .string "smaller"
.bigger: .string "bigger"
.global main
.type main,@function

main:
    pushq %rbp
    mov %rsp, %rbp
    pushq $1
    pushq $1
    movq $10, -8(%rbp)
    movq $5, -16(%rbp)
    mov -16(%rbp), %rcx
    mov $0, %r10
    cmp -8(%rbp), %rcx
    cmovnb .bool_test(%rip), %r10
    mov %r10, %r9
    cmp $1, %r9
    jne .jmpif14
    mov .bigger(%rip), %rsi
    mov $0, %rax
    leaq .str_wformat(%rip), %rdi
    call printf
.jmpif14:
    mov .smaller(%rip), %rsi
    mov $0, %rax
    leaq .str_wformat(%rip), %rdi
    call printf
    leave
    movq $0, %rax
    ret
    leave
    ret
