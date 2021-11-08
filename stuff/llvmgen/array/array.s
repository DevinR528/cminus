	.text
	.file	"array"
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$.Lfmtstr, %edi
	movl	$9, %esi
	xorl	%eax, %eax
	callq	printf
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc

	.type	.Lfmtstr,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lfmtstr:
	.asciz	"%d\n"
	.size	.Lfmtstr, 4

	.section	".note.GNU-stack","",@progbits
