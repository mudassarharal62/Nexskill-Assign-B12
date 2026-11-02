# Check if a value exists in a dictionary

my_dict = {
    "name": "Umair",
    "age": 22,
    "city": "Yazman",
    "course": "Full Stack AI",
    "cgpa": 3.75,
    "is_student": True
}

if 22 in my_dict.values():
    print("Exist")
else:
    print("Does not exist")