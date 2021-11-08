
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
    pushq $1
    pushq $1
    pushq $1
    movq $100, -32(%rbp)
    movq $10, -8(%rbp)
    movq $5, -24(%rbp)
    mov -8(%rbp), %rdx
    mov -32(%rbp), %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r8
    mov -24(%rbp), %rdx
    mov %r8, %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r9
    movq %r9, -16(%rbp)
    mov -16(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    movq $2, %rdx
    mov -32(%rbp), %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r8
    movq $2, %rdx
    mov %r8, %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r9
    movq $5, %rdx
    mov %r9, %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r8
    mov %r8, %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    mov -24(%rbp), %rdx
    mov $30, %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r9
    mov %r9, %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    mov -16(%rbp), %rdx
    mov -32(%rbp), %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r8
    mov -16(%rbp), %rdx
    mov %r8, %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r9
    movq $5, %rdx
    mov %r9, %rax
    cdq
    idiv %rdx
    movq %rax, %rdx
    mov %rdx, %r8
    movq %r8, -40(%rbp)
    mov -40(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    movq $0, %rax
    leave
    ret
