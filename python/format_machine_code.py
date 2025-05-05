import struct

# Function to convert vector assembly instructions to machine code (dummy function for illustration)
def convert_vector_to_machine_code(instruction):
    # For this example, we assume we have a direct mapping for simplicity.
    # A real implementation would require an assembler or specific machine code generation process.
    machine_code_mapping = {
        'vetvli ra, zero, e32,m1,tu,mu': '100070D7',
        'vadd.vi v1, v0, 0x5': '0202B0D7',
        'vadd.vi v2, v1, 0x6': '02133157'
    }
    
    return machine_code_mapping.get(instruction, '00000000')  # Return '00000000' if instruction not found

# Function to reverse the byte order of the machine code
def reverse_byte_order(machine_code):
    # Split the machine code into bytes and reverse the byte order
    bytes_list = [machine_code[i:i+2] for i in range(0, len(machine_code), 2)]
    reversed_bytes = bytes_list[::-1]  # Reverse the list of bytes
    return ''.join(reversed_bytes)

# Function to convert vector code to machine code and reverse the byte order
def process_vector_code(vector_code):
    # Convert vector assembly code to machine code
    machine_codes = [convert_vector_to_machine_code(instruction) for instruction in vector_code]
    
    # Reverse the byte order for each machine code
    reversed_machine_codes = [reverse_byte_order(code) for code in machine_codes]
    
    return reversed_machine_codes

# Example vector code (vector assembly instructions)
vector_code = [
    'vetvli ra, zero, e32,m1,tu,mu',
    'vadd.vi v1, v0, 0x5',
    'vadd.vi v2, v1, 0x6'
]

# Process the vector code
reversed_machine_codes = process_vector_code(vector_code)

# Output the results
for i, reversed_code in enumerate(reversed_machine_codes):
    print(f'Original Machine Code for instruction {i+1}: {convert_vector_to_machine_code(vector_code[i])}')
    print(f'Reversed Byte Order: {reversed_code}')
    print()

