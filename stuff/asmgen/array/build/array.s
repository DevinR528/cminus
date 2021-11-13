
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
    pushq                      $0                             # 
    pushq                      $0                             # 
    pushq                      $0                             # 
    pushq                      $0                             # 
    movq                       $0,           -40(%rbp)
    movq                       $0,           -32(%rbp)
    movq                       $0,           -24(%rbp)
    movq                       $0,           -16(%rbp)
    movq                       $0,            -8(%rbp)
    movq                       $0,           -32(%rbp)
    movq                       $9,           -24(%rbp)
    movq                      $11,           -16(%rbp)
    movq                      $15,            -8(%rbp)
    mov                 -32(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -24(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -16(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                  -8(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    pushq                      $0                             # 
    pushq                      $0                             # 
    pushq                      $0                             # 
    pushq                      $0                             # 
    pushq                      $0                             # 
    movq                       $0,           -80(%rbp)
    movq                      $16,           -72(%rbp)
    movq                      $12,           -64(%rbp)
    movq                      $10,           -56(%rbp)
    movq                       $8,           -48(%rbp)
    mov                 -72(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -64(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -56(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    mov                 -48(%rbp),                %rsi        # 
    mov                        $0,                %rax        # 
    leaq       .int_wformat(%rip),                %rdi
    call                  printf
    movq                       $0,                %rax
    leave
    ret
