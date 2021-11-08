
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
    pushq                    %rbp                             # 
    mov                      %rsp,                %rbp        # 
    pushq                      $1                             # 
    pushq                      $1                             # 
    pushq                      $1                             # 
    pushq                      $1                             # 
    pushq                      $1                             # 
    movq                       $3,           -32(%rbp)
    movq                      $10,            -8(%rbp)
    movq                       $5,           -24(%rbp)
    mov                 -24(%rbp),                %r13        # 
    imul                 -8(%rbp),                %r13
    sub                       $3,                %r13
    movq                     %r13,           -16(%rbp)
    mov                 -16(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -24(%rbp),                %r13        # 
    imul                 -8(%rbp),                %r13
    sub                -16(%rbp),                %r13
    mov                      %r13,                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    movq                       $3,                %rax
    sub                -24(%rbp),                %rax
    mov                      %rax,                %r13        # we had to spill a printf register
    mov                      %r13,                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -24(%rbp),                %r14        # 
    add                       $4,                %r14
    movq                     %r14,           -40(%rbp)
    mov                 -40(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    movq                       $0,                %rax
    leave
    ret
