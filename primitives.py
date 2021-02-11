#!/usr/bin/env python3
import random
import regex
import string


def identity(left):
    return left


def concatenate(left, right):
    return left + right


def optional(left):
    return left if left.endswith("?") else f"{left}?"


def range(left, right):
    if left == right:
        return regex.escape(chr(left))
    return f"[{regex.escape(chr(min(left, right)))}-{regex.escape(chr(max(left, right)))}]"


def negatedRange(left, right):
    # if left == right:
    #    return f"[^{regex.escape(chr(left))}]"
    return f"[^{regex.escape(chr(min(left, right)))}-{regex.escape(chr(max(left, right)))}\\n\\r]"


def randomPrintableAsciiCode():
    return random.randint(32, 126)


def randomCharacter():
    return regex.escape(chr(random.randint(32, 126)))


# def whitespace():
#    return "([\s]+)"


def wildcard():
    return "."
