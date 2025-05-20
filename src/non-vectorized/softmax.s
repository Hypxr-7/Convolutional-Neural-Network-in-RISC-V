.data
Vals:
.float -7.942513 -12.333942 -3.000695 3.169582 -14.738624 7.824346 -11.468449 -4.637082 -2.740747 -3.158628 

Out:
.float 0 0 0 0 0 0 0 0 0 0 

.text

.globl main

main:
    la a0, Vals
    la a1, Out
    
    li t1, 1    # i = 1 | f1 will already contain one value when entering loop for max
    li t2, 10   # 10 elements
    
    flw f1, 0(a0)
max_loop:
    beq t1, t2, end_max_loop
    slli t1, t1, 2
    add s0, t1, a0
    flw f2, 0(s0)   # load input[i]
    
    fmax.s f1, f1, f2
    
    srli t1, t1, 2
    addi t1, t1, 1
    j max_loop
end_max_loop: # f1 now contains the max element
    
    fcvt.s.w f0, zero   # f0 contains sum | init to 0
    
    li t1, 0    #  i = 0

approx_loop:
    beq t1, t2, end_approx_loop
    slli t1, t1, 2
    add s0, a0, t1
    
    flw f2, 0(s0)
    
    fsub.s f2, f2, f1
    
    li t3, 1    # j = 1
    li t4, 6    # do the loop 5 times
    li t5, 1    # contains 1
    fcvt.s.w f3, t5 # term = 1.0
    fcvt.s.w f4, t5 # result = 1.0
    
    fcvt.s.w f7, zero
    flt.s t6, f2, f7    # t6 1 if f2 < 0 | current number is neative
    beq t6, t3,  negative_num_case
    j exp_calc
 
 negative_num_case:
    li t5, -1
    fcvt.s.w f7, t5 # load -1.0
    fmul.s f2, f2, f7
    
 exp_calc:
    beq t3, t4, end_exp_calc
    fcvt.s.w f5, t3 
    fdiv.s f6, f2, f5   # f6 = x / i
    fmul.s f3, f3, f6   # term *= f6
    
    fadd.s f4, f4, f3   # result += term
    
    addi t3, t3, 1
    j exp_calc
end_exp_calc:
    beqz t6, skip_neg_handle
    li t3, 1
    fcvt.s.w f7, t3 # load 1.0
    fdiv.s f4, f7, f4

skip_neg_handle:
    fadd.s f0, f0, f4
    
    fsw f4, 0(s0)
    
    srli t1, t1, 2
    addi t1, t1, 1
    j approx_loop
end_approx_loop:
    li t1, 0    # i = 0
    
normalize:
    beq t1, t2, end_normalize
    slli t1, t1, 2
    
    add s0, t1, a0
    add s1, t1, a1
    
    flw f1, 0(s0)
    
    fdiv.s f1, f1, f0
    
    fsw f1, 0(s1)
    
    srli t1, t1, 2
    addi t1, t1, 1
    j normalize
end_normalize:

end_softmax:
    li a0, 0
    ret
    
    


    
    
    
    
    
    
    
    