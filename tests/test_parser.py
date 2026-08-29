from parsing.python_parser import parse_python_code


code = """
import os


class UserService:

    def login(self):
        validate_user()
        self.create_session()
        return True

    def create_session(self):
        print("session created")


def validate_user():
    check_password()


def check_password():
    return True
"""


result = parse_python_code(code)

print(result)