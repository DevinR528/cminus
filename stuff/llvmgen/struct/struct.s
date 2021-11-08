	.text
	.file	"struct"
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5
.LCPI0_0:
	.quad	0
	.quad	0
	.quad	1
	.quad	2
.LCPI0_1:
	.quad	0
	.quad	1
	.quad	2
	.quad	3
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
	.cfi_startproc
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	vmovaps	.LCPI0_0(%rip), %ymm0
	vmovaps	.LCPI0_1(%rip), %ymm1
	movl	$.Lfmtstr, %edi
	movl	$10, %esi
	xorl	%eax, %eax
	vmovups	%ymm0, (%rsp)
	movq	$10, (%rsp)
	vmovups	%ymm1, 48(%rsp)
	movq	$3, 32(%rsp)
	movq	$4, 80(%rsp)
	movq	$4, 40(%rsp)
	vzeroupper
	callq	printf
	movl	$.Lfmtstr.1, %edi
	xorl	%esi, %esi
	xorl	%eax, %eax
	callq	printf
	movl	$.Lfmtstr.2, %edi
	movl	$2, %esi
	xorl	%eax, %eax
	callq	printf
	movl	$.Lfmtstr.3, %edi
	movl	$4, %esi
	xorl	%eax, %eax
	callq	printf
	addq	$88, %rsp
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

	.type	.Lfmtstr.1,@object
.Lfmtstr.1:
	.asciz	"%d\n"
	.size	.Lfmtstr.1, 4

	.type	.Lfmtstr.2,@object
.Lfmtstr.2:
	.asciz	"%d\n"
	.size	.Lfmtstr.2, 4

	.type	.Lfmtstr.3,@object
.Lfmtstr.3:
	.asciz	"%d\n"
	.size	.Lfmtstr.3, 4

	.section	".note.GNU-stack","",@progbits
