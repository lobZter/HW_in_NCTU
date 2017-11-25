
file_path = raw_input("File path: ")
a = input("intial a: ")
b = input("intial b: ")

with open(file_path.strip()) as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]

for line in lines:
    N, m, p = observe(line)
    print 'MLE: {} | Prior a: {}, b: {} |'.format(p, a, b),
    # a = m + a
    # b = N - m + b
    a += m
    b += (N - m)
    print 'Posterior a: {}, b: {}'.format(a, b)

def observe(input):
    head_count = 0
    tail_count = 0
    total_count = len(input)
    while input != "":
        if input[0] == "1":
            head_count += 1
        if input[0] == "0":
            tail_count += 1
        input = input[1:]
    return total_count, head_count, float(head_count)/total_count
