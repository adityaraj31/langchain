from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    is_student: bool

new_person: Person = {
    "name": "Alice",
    "age": 20,
    "is_student": True
}

print(new_person)