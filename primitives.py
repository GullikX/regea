#!/usr/bin/env python3
import random
import string


def concatenate(left, right):
    return left + right


def regexRange(left, right):
    if len(left) != 1 or len(right) != 1:
        raise ValueError
    if left == "." or right == ".":
        return "."
    if left.islower() and not right.islower():
        raise ValueError
    if left.isupper() and not right.isupper():
        raise ValueError
    if left.isdecimal and not right.isdecimal():
        raise ValueError
    return f"[{left}-{right}]"


def regexOr(left, right):
    if len(left) != 1 or len(right) != 1:
        raise ValueError
    if left == "." or right == ".":
        return "."
    return f"[{left}|{right}]"


def randomLowercaseLetter():
    return random.choice(string.ascii_lowercase)


def randomUppercaseLetter():
    return random.choice(string.ascii_uppercase)


def randomDigit():
    return random.choice(string.digits)


def wildcard():
    return "."
