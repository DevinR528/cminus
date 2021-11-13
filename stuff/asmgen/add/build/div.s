
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
    movq                     $100,            -8(%rbp)
    pushq                      $0                             # 
    movq                      $10,           -16(%rbp)
    pushq                      $0                             # 
    movq                       $5,           -24(%rbp)
    pushq                      $0                             # 
    mov                 -16(%rbp),                %rax        # 
    pushq                    %rax                             # rax used
    popq                     %r13                             # use new reg for rax contents right
    mov                  -8(%rbp),                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r13
    movq                     %rax,                %r13
    mov                 -24(%rbp),                %rax        # 
    pushq                    %rax                             # rax used
    popq                     %r10                             # use new reg for rax contents right
    mov                      %r13,                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r10
    movq                     %rax,                %r10
    movq                     %r10,           -32(%rbp)
    mov                 -32(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    movq                       $2,                %rax
    pushq                    %rax                             # rax used
    popq                     %r13                             # use new reg for rax contents right
    mov                  -8(%rbp),                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r13
    movq                     %rax,                %r13
    movq                       $2,                %rax
    pushq                    %rax                             # rax used
    popq                     %r10                             # use new reg for rax contents right
    mov                      %r13,                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r10
    movq                     %rax,                %r10
    movq                       $5,                %rax
    pushq                    %rax                             # rax used
    popq                     %r13                             # use new reg for rax contents right
    mov                      %r10,                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r13
    movq                     %rax,                %r13
    mov                      %r13,                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -24(%rbp),                %rax        # 
    pushq                    %rax                             # rax used
    popq                     %r10                             # use new reg for rax contents right
    mov                       $30,                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r10
    movq                     %rax,                %r10
    mov                      %r10,                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    pushq                      $0                             # 
    mov                 -32(%rbp),                %rax        # 
    pushq                    %rax                             # rax used
    popq                     %r13                             # use new reg for rax contents right
    mov                  -8(%rbp),                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r13
    movq                     %rax,                %r13
    mov                 -32(%rbp),                %rax        # 
    pushq                    %rax                             # rax used
    popq                     %r10                             # use new reg for rax contents right
    mov                      %r13,                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r10
    movq                     %rax,                %r10
    movq                       $5,                %rax
    pushq                    %rax                             # rax used
    popq                     %r13                             # use new reg for rax contents right
    mov                      %r10,                %rax        # move lhs to dividend `rdx:rax / whatever`
    cdq
    idiv                     %r13
    movq                     %rax,                %r13
    movq                     %r13,           -40(%rbp)
    mov                 -40(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    movq                       $0,                %rax
    leave
    ret
