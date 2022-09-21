from random import randint


def predicted_label(x):
    if randint(0, 100) > 50:
        return "normal"
    else:
        return "pneumonia"
