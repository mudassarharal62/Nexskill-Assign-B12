# Python Program to Add a Key-Value Pair to the Dictionary. The program takes a key-value pair 
# and adds it to the dictionary.

d = {
    "name": "Umair",
    "age": 21,
    "city": "Yazman"
}

print(d)

key = input("Enter key: ")

value = input("Enter value: ")

d.update({key:value})

print(d)