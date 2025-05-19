.data
Matrix:
.float 4 4 5 5 2 2 2 0 3 7 3 8 6 1 4 3 1 0 3 4 6 7 2 4
.float 2 3 6 4 2 3 9 4 6 2 2 2 1 9 7 1 4 0 6 3 5 1 7 1
.float 9 2 7 0 4 7 7 2 4 4 8 2 3 5 3 1 2 8 5 0 4 7 2 3
.float 4 4 3 1 0 7 6 4 7 2 3 2 5 3 6 5 3 5 1 7 0 6 2 6
.float 2 3 8 3 3 6 7 8 7 1 1 3 4 0 1 0 7 6 0 1 3 2 4 5
.float 1 9 0 4 5 5 2 7 5 0 6 6 5 3 3 8 5 2 9 7 1 6 6 9
.float 3 5 6 8 0 5 7 3 5 2 9 1 1 0 2 0 6 3 5 3 2 4 7 5
.float 5 2 4 7 6 8 2 2 9 0 3 2 4 1 2 1 3 1 4 1 6 0 3 2
.float 8 9 1 7 4 4 4 1 3 0 7 8 0 7 3 3 7 9 2 7 1 3 3 4
.float 7 4 3 2 6 2 4 6 3 7 3 2 7 0 4 1 7 6 1 1 2 3 6 3
.float 1 1 3 2 5 4 3 1 1 6 0 5 7 6 2 1 3 3 5 2 6 1 2 8
.float 6 7 7 3 8 0 2 4 0 9 0 2 7 7 1 5 4 4 5 6 5 1 7 9
.float 1 0 8 1 6 7 7 9 4 0 3 7 2 0 3 5 2 6 3 6 1 5 8 6
.float 9 2 2 6 7 4 6 3 8 1 0 8 3 1 5 4 3 0 9 1 3 1 0 6
.float 0 4 2 3 2 4 2 3 2 4 2 9 7 1 1 8 0 5 3 9 5 6 5 5
.float 7 2 2 0 7 7 8 7 6 1 4 3 5 3 4 1 5 3 5 5 4 0 1 2
.float 6 1 1 5 3 7 5 0 6 4 4 6 6 1 7 2 2 4 7 6 1 2 8 4
.float 6 3 6 3 5 7 2 4 3 6 0 4 8 9 4 2 1 2 5 4 3 6 3 2
.float 4 4 6 3 1 5 7 7 7 4 4 2 3 4 4 6 1 8 4 5 2 6 7 1
.float 4 8 6 6 1 2 6 1 2 6 2 1 3 3 4 6 6 7 2 4 1 3 7 2
.float 2 3 4 5 1 6 3 7 4 2 6 3 8 5 9 6 3 3 5 3 2 5 2 4
.float 0 5 2 4 3 1 2 6 4 2 3 6 7 7 2 8 3 2 4 3 6 6 4 8
.float 4 4 7 1 3 5 3 4 5 2 3 6 2 5 5 5 5 6 8 2 3 4 6 4
.float 6 3 1 4 8 1 6 5 3 4 5 4 6 3 8 7 2 6 7 5 3 1 3 1

Maxpooled:
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0 
.float 0 0 0 0 0 0 0 0 0 0 0 0
.float 0 0 0 0 0 0 0 0 0 0 0 0
.float 0 0 0 0 0 0 0 0 0 0 0 0

.text

.globl main

main:
    la a0, Matrix
    la a1, Maxpooled
    
    li s0, 576 # elements in input matrix
    li s1, 144 # elements in output matrix
    li s2, 24 # input dim
    li s3, 12 # output dim
  
    li t0, 0 # k = 0
 
loop:
    beq t0, s1, end
    
    # mapping 1D array to 2D
    div t1, t0, s3 # t1 = i
    rem t2, t0, s3 # t2 = j
    
    add t3, t1, t1 # temp1 = 2 * i
    addi t4, t3, 1 # temp2 = 2 * i + 1
    add t5, t2, t2 # temp3 = 2 * j
    addi t6, t5, 1 # temp4 = 2 * j + 1
    
    mul s4, t3, s2 # temp1 * input_dim
    add t1, s4, t5 # top left 
    add t2, s4, t6 # top right
    
    mul s4, t4, s2 # temp2* input_dim
    add t3, s4, t5 # bottom left
    add t4, s4, t6 # bottom right
    
    slli t1, t1, 2 # compute amount to offset
    slli t2, t2, 2
    slli t3, t3, 2
    slli t4, t4, 2
    
    add t1, a0, t1 # add offset
    add t2, a0, t2
    add t3, a0, t3
    add t4, a0, t4
    
    flw f0, 0(t1) # load values
    flw f1, 0(t2)
    flw f2, 0(t3)
    flw f3, 0(t4)
    
    fmax.s f0, f0, f1 # get max
    fmax.s f0, f0, f2
    fmax.s f0, f0, f3
    
    slli t0, t0, 2 # i = i * 4 | we will divide by 4 for proper incrementing | done to save registers
    
    add t1, a1, t0
    
    fsw f0, 0(t1)
    
    srli t0, t0, 2 # i = i / 4
    
    addi t0, t0, 1
    
    j loop

end:
    li a0, 0
    ret
