def format_float_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip().startswith('.float'):
                parts = line.strip().split()
                keyword = parts[0]
                numbers = ', '.join(parts[1:])
                formatted_line = f"{keyword} {numbers}\n"
                outfile.write(formatted_line)
            else:
                outfile.write(line)



format_float_lines('weight1.txt', 'weight.txt')
