import json
# Load the previously saved JSON data from the file
with open('data/转换后的数据.json', 'r', encoding='utf-8') as json_file:
    transformed_data = json.load(json_file)

# Initialize a variable to hold the maximum length
max_length = 0

# Iterate over the data to find the maximum length of the "output" field
for data in transformed_data:
    output_length = len(data["output"])
    if output_length > max_length:
        max_length = output_length
        print(data["output"])

print(max_length)
