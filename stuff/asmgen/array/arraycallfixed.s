
.text

.char_wformat: .string "%c\n"
.int_wformat: .string "%d\n"
.float_wformat: .string "%f\n"
.str_wformat: .string "%s\n"
.char_rformat: .string "%c"
.int_rformat: .string "%d"
.float_rformat: .string "%f"

.global set_idx
.type set_idx,@function

set_idx:
    pushq %rbp
    mov %rsp, %rbp

    pushq %rdi # -8 the index value
    pushq %rsi # -16 the array pointer

    movq -16(%rbp), %rdx
    movq -8(%rbp), %rbx
    addq %rdx, %rbx
    movq (%rbx), %rax

    leave
    ret

.global main
.type main,@function

main:
    pushq %rbp
    mov %rsp, %rbp

    pushq $1

    pushq $1
    pushq $1
    pushq $1

    movq $2, -32(%rbp)
    movq $1, -24(%rbp)
    movq $3, -16(%rbp)

    movq $2, %rdi
    leaq -32(%rbp), %rax
    movq %rax, %rsi
    call set_idx

    movq %rax, -8(%rbp)
    mov -8(%rbp), %rsi
    mov $0, %rax
    leaq .int_wformat(%rip), %rdi
    call printf
    leave
    ret
