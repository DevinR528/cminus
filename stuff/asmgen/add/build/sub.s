
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
    pushq                      $0                             # 
    movq                       $3,            -8(%rbp)
    pushq                      $0                             # 
    movq                      $10,           -16(%rbp)
    pushq                      $0                             # 
    movq                       $5,           -24(%rbp)
    pushq                      $0                             # 
    mov                 -24(%rbp),                %rax        # 
    imul                -16(%rbp),                %rax
    sub                       $3,                %rax
    movq                     %rax,           -32(%rbp)
    mov                 -32(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -24(%rbp),                %rax        # 
    imul                -16(%rbp),                %rax
    sub                -32(%rbp),                %rax
    mov                      %rax,                %r13        # we had to spill a printf register
    mov                      %r13,                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    movq                       $3,                %r10
    sub                -24(%rbp),                %r10
    mov                      %r10,                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    pushq                      $0                             # 
    mov                 -24(%rbp),                %rax        # 
    add                       $4,                %rax
    movq                     %rax,           -40(%rbp)
    mov                 -40(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    movq                       $0,                %rax
    leave
    ret
