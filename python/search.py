
import struct
import numpy as np

def hex_to_float(hex_val):
    return struct.unpack('!f', bytes.fromhex(hex_val))[0]

def extract_floats_from_log(filename, instruction="vlse32", first_only=False):
    floats = []
    with open(filename, 'r') as file:
        for line in file:
            if instruction in line:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue  # Not a valid line
                hex_string = parts[6]

                # Split into 32-bit float chunks
                chunks = [hex_string[i:i+8] for i in range(0, len(hex_string), 8)]
                chunks.reverse()  # Correct memory order

                try:
                    if first_only:
                        floats.append(hex_to_float(chunks[0]))
                    else:
                        for chunk in chunks:
                            floats.append(hex_to_float(chunk))
                except Exception as e:
                    print(f"Skipping invalid hex: {e}")
    return floats

def display_floats(values, shape=None, channel_view=False):
    if shape is not None:
        try:
            array = np.array(values, dtype=np.float32).reshape(shape)
        except ValueError as e:
            raise ValueError(f"Could not reshape: {e}")

        if channel_view and len(shape) == 3:
            channels, height, width = shape
            for c in range(channels):
                print(f"\nChannel {c}:")
                for i in range(height):
                    row = " ".join(f"{array[c][i][j]:8.6f}" for j in range(width))
                    print(row)
        else:
            print(array)
    else:
        print("Flat output:")
        for value in values:
            print(f"{value:.6f}", end=" ")
        print()

if __name__ == "__main__":
    filename = "build/logs/conv.txt"
    instruction = "fsw      ft0, 0x0(a1)"  # Change to "vlse32" or "vfredmax" as needed
    
    values = extract_floats_from_log(filename, instruction=instruction, first_only=(instruction == "vfredosum"))

    # # get the last 4608 values
    # print(len(values), "values before slicing")
    # values = values[-16:-6]
    # print(len(values), "values extracted")
    # values = values[:10]


    # Final Layer Outputs:
    #   - For Con2d, set shape = (8,24,24), set channel_view = True
    #   - For ReLU, set shape = (8,24,24), set channel_view = True
    #   - For Max Pool, set shape = (8,12,12), set channel_view = True
    #   - For Flatten, set shape = None, set channel_view = False
    #   - For Dense, set shape = None, set channel_view = False
    #   - For Softmax, set shape = None, set channel_view = False
    display_floats(
        values,
        shape=None,              # e.g., (8, 24, 24) if reshaping
        channel_view=False       # Set True if using (C, H, W)
    )

