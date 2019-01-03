#!/usr/bin/python3

import random
import sys

with sys.stdin as f_in:
    for line in f_in:
        print([random.random() for _ in range(5)])
