table_width = 15


def get_float_fmt(width, column):
    return ' | '.join([f'{{:{width}}}'] + [f'{{:{width}.4f}}'] * column)


def get_string_fmt(width, column):
    return ' | '.join([f'{{:{width}}}'] + [f'{{:{width}}}'] * column)


def get_separator(fmt, width, column):
    return fmt.format(*['-' * width for _ in [width] * (column + 1 + 4)])