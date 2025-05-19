# create_float_file.py

def create_float_file(filename):
    value = "1"
    count_per_line = 12
    lines = 12

    with open(filename, "w") as file:
        line = ".float " + " ".join([value] * count_per_line) + "\n"
        for _ in range(lines):
            file.write(line)

if __name__ == "__main__":
    create_float_file("output.txt")
