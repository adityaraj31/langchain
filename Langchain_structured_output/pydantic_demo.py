from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'nitish'
    age: Optional[int] = None
    email: EmailStr   
    cgpq: float = Field(gt=0, le=10, default=5, description="CGPA must be between 0 and 10")


new_student = {'name': 'John', 'age': 20, 'email': 'xyz@gmail.com', 'cgpq': 8.5}

Student = Student(**new_student)

student_dict = dict(Student)

print(student_dict['age'])

student_json = Student.model_dump_json()

print(student_json)