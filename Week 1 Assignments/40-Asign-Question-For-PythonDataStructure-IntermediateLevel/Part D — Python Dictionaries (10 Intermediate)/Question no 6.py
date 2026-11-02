# 6. Given a nested dictionary, safely access a deeply nested key.

nestedDict = {
    101: {
        "name": "Ayesha",
        "age": 20,
        "city": "lahore"
    },
    102: {
        "name": "Umair",
        "age": 22,
        "city": "Yazman"
    },
}

print(f"Nested dictionary: {nestedDict}")

print(f"Safely access a deeply nested key 'name': {nestedDict[101]['name']}")

print(f"Safely access a deeply nested key 'name' using get(): {nestedDict.get(101).get('name')}")