#!/usr/bin/env python3
import random
import string


def concatenate(left, right):
    return left + right


def optional(left):
    return left if left.endswith("?") else f"{left}?"


def lowercaseLetter():
    return "[a-z]"


def uppercaseLetter():
    return "[A-Z]"


def digit():
    return "[0-9]"


def wildcard():
    return "."
