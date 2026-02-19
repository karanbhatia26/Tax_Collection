import ast

def sum_list(numbers):
    return sum(numbers)

if __name__ == "__main__":
    data = ast.literal_eval(input().strip())
    print(sum_list(data))