#!/usr/bin/env python3
import random
import regex
import string


def concatenate(left, right):
    return left + right


def optional(left):
    return left if left.endswith("?") else f"{left}?"


def lowercaseLetter():
    return random.choice(string.ascii_lowercase)


def anyLowercaseLetter():
    return "[a-z]"


def uppercaseLetter():
    return random.choice(string.ascii_uppercase)


def anyUppercaseLetter():
    return "[A-Z]"


def digit():
    return random.choice(string.digits)


def anyDigit():
    return "[0-9]"


def whitespace():
    return "([\s]+)"


def specialCharacter():
    return regex.escape(random.choice(string.punctuation))


def anySpecialCharacter():
    return "[^a-zA-Z0-9\s]"


def wildcard():
    return "."
