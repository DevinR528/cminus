
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
    movq                       $3,            -8(%rbp)
    pushq                      $1                             # 
    movq                      $10,           -16(%rbp)
    pushq                      $1                             # 
    movq                       $5,           -24(%rbp)
    pushq                      $1                             # 
    mov                 -24(%rbp),                %rax        # 
    imul                -16(%rbp),                %rax
    sub                       $3,                %rax
    movq                     %rax,           -32(%rbp)
    movq                       $0,                %rax
    leave
    ret
