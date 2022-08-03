from __future__ import division, absolute_import, print_function
from .misctools import to_decimals, int_else_float
from numbers import Real, Complex

import re
import numpy as np


def list2complex(thing):
    if not isinstance(thing, list):
        raise ValueError("list2complex expecting list")

    if len(thing) == 1:
        return complex(thing[0])

    if len(thing) == 2:
        return complex(thing[0], thing[1])

    raise ValueError("expecting list with 1 or 2 elements in list2complex")


def matrix_to_string(tf, decimals=None):
    """
    Take an numpy array and returns the correspinding transform string.

    Returns the empty string for the identity matrix.
    """
    def ze_formatter(i, j):
        return to_decimals(tf[i, j], decimals)

    if not isinstance(tf, np.ndarray) or not tf.shape == (3, 3):
        raise ValueError

    if tf[0, 1] == 0 and tf[1, 0] == 0:
        if tf[1, 1] != tf[0, 0]:
            scale = f'scale({ze_formatter(0, 0)},{ze_formatter(1, 1)})'

        elif tf[0, 0] != 1:
            scale = f'scale({ze_formatter(0, 0)})'

        else:
            scale = ''

        if tf[1, 2] != 0:
            translate = f'translate({ze_formatter(0, 2)} {ze_formatter(1, 2)})'

        elif tf[0, 2] != 0:
            translate = f'translate({ze_formatter(0, 2)})'

        else:
            translate = ''

        return translate + scale

    return (
        'matrix(' +
        ze_formatter(0, 0) + ' ' +
        ze_formatter(1, 0) + ' ' +
        ze_formatter(0, 1) + ' ' +
        ze_formatter(1, 1) + ' ' +
        ze_formatter(0, 2) + ' ' +
        ze_formatter(1, 2) + ')'
    )


def is_svg_matrix(a):
    assert isinstance(a, np.ndarray)
    return a.shape == (3, 3) and \
        a[2, 0] == a[2, 1] == 0 and \
        a[2, 2] == 1


def generate_transform_if_not_already_string(*args, html_style=False):
    if len(args) == 1 and isinstance(args[0], str):
        return args[0]
    return generate_transform(*args, html_style=html_style)


def number_list_2_ordinary_vec(values):
    assert isinstance(values, list)
    assert 1 <= len(values) <= 2
    assert all(isinstance(x, Real) for x in values)

    if len(values) == 1:
        return np.array([[values[0]], [0]])

    if len(values) == 2:
        return np.array([[values[0]], [values[1]]])

    assert False


def ordinary_vec_2_list(vec):
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (2, 1)
    return [vec[0, 0], vec[1, 0]]


def arrays_are_close(a, b):
    return all(y for x in np.isclose(a, b) for y in x)


def __normalize_transform_translation_rightmost(*args):
    pairs = [{'type': command, 'values': values} for command, values in string_and_values_iterator(args)]
    translation = 0
    while any(x['type'] == 'translate' for x in pairs):
        if pairs[-1]['type'] == 'translate':
            translation += list2complex(pairs[-1]['values'])
            pairs.pop()
            continue

        for i in range(len(pairs) - 2, -1, -1):
            p = pairs[i]
            if p['type'] != 'translate':
                continue
            q = pairs[i + 1]
            # PQ = QP' where P and P' are translations
            # PQ(x) = Q(x) + z (definition of z)
            # QP'(x) = Q(x + w) = Q(x) + Q(w)
            # Q(x) + z = Q(x) + Q(w)
            # w = Qinv(z)
            Q3d = parse_transform(q['type'], q['values'])
            if not is_svg_matrix(Q3d):
                raise ValueError
            Q = Q3d[0:2, 0:2]
            Qinv = np.linalg.inv(Q)

            z = number_list_2_ordinary_vec(p['values'])
            w = Qinv.dot(z)

            # the check
            P3d = parse_transform(p['type'], p['values'])
            Pprime3d = parse_transform('translate', w)
            assert arrays_are_close(P3d.dot(Q3d), Q3d.dot(Pprime3d))

            # move Q to position i
            p['type'] = q['type']
            p['values'] = q['values']

            # redefine position i + 1 to be translation
            q['type'] = 'translate'
            q['values'] = ordinary_vec_2_list(w)
            if q['values'][1] == 0:
                q['values'].pop()

            # (and i guess one can continue?)
            assert pairs[i]['type'] != 'translate'

    tokens_prefix = []
    for p in pairs:
        tokens_prefix.append(p['type'])
        tokens_prefix.append(p['values'])

    assert 'translate' not in tokens_prefix

    return tokens_prefix, translation


def normalize_transform_translation_rightmost(*args):
    tokens, translation = __normalize_transform_translation_rightmost(*args)
    if translation != 0:
        tokens.append('translate')
        tokens.append(translation)
    return generate_transform(tokens)


def compound_translations(*args):
    pairs = [{'type': command, 'values': values} for command, values in string_and_values_iterator(args)]
    for i in range(len(pairs) - 2, -1, -1):
        p = pairs[i]
        if p['type'] != 'translate':
            continue
        q = pairs[i + 1]
        if q['type'] == 'translate':
            q_vals = q['values']
            p_vals = p['values']
            assert 1 <= len(q_vals) <= 2 and 1 <= len(p_vals) <= 2
            x = p_vals[0] + q_vals[0]
            q_y = q_vals[1] if len(q_vals) == 2 else 0
            p_y = p_vals[1] if len(p_vals) == 2 else 0
            y = p_y + q_y
            q['type'] = 'skip me Im dead'
            p['values'] = [x, y] if y != 0 else [x]

    to_return = []
    for p in pairs:
        if p['type'] == 'skip me Im dead':
            continue
        to_return.append(p['type'])
        to_return.extend(p['values'])

    return to_return


# =============================================
# start beginning of new stuff
# =============================================


allowable = {
    'translate': [1, 2],
    'rotate': [1, 3],
    'scale': [1, 2],
    'skewX': [1],
    'skewY': [1],
    'matrix': [6],
}


def command_and_values_to_string(command, values, decimals=None, html_style=False):
    """
    note: does ***not*** attempt to return an empty string for an identity transform
    """
    assert decimals is None or isinstance(decimals, int)
    assert command in allowable
    assert len(values) in allowable[command]
    assert isinstance(values, list)
    assert all(isinstance(v, Real) for v in values)

    units = None

    def ze_formatter(num):
        prefix = to_decimals(num, decimals)
        return prefix + units if html_style else prefix

    if 'matrix' in command:
        assert not html_style
        return 'matrix(' + ','.join(list(map(ze_formatter, values))) + ')'

    elif 'translate' in command:
        units = 'px'
        if all(x == 0 for x in values):
            return ''
        if len(values) == 2 and values[1] == 0:
            values.pop()
        if len(values) == 1 and values[0] == 0:
            return ''
        return 'translate(' + ','.join(list(map(ze_formatter, values))) + ')'

    elif 'scale' in command:
        units = ''
        return 'scale(' + ','.join(list(map(ze_formatter, values))) + ')'

    elif 'rotate' in command:
        units = 'deg'
        if float(values[0]) == 0:
            return ''
        return 'rotate(' + ','.join(list(map(ze_formatter, values))) + ')'

    elif 'skewX' in command:
        assert not html_style
        return 'skewX(' + ','.join(list(map(ze_formatter, values))) + ')'

    elif 'skewY' in command:
        assert not html_style
        return 'skewY(' + ','.join(list(map(ze_formatter, values))) + ')'

    print('Unknown SVG transform type:', command)
    assert False


def command_and_values_to_matrix(command, values):
    assert isinstance(command, str)
    assert all(isinstance(v, Real) for v in values)
    assert command in allowable
    assert len(values) in allowable[command]

    transform = np.identity(3)

    if command == 'matrix':
        transform[0:2, 0:3] = np.array([values[0:6:2], values[1:6:2]])

    elif command == 'translate':
        transform[0, 2] = values[0]
        if len(values) == 2:
            transform[1, 2] = values[1]

    elif 'scale' in command:
        x = values[0]
        y = values[1] if (len(values) == 2) else x
        transform[0, 0] = x
        transform[1, 1] = y

    elif 'rotate' in command:
        if len(values) == 3:
            offset = values[1:3]

        else:
            offset = (0, 0)

        tf_offset = np.identity(3)
        tf_offset[0:2, 2:3] = np.array([[offset[0]], [offset[1]]])

        tf_rotate = np.identity(3)

        degrees = values[0] % 360
        assert 0 <= degrees < 360

        if degrees == 90:
            tf_rotate[0:2, 0:2] = np.array([[0, -1], [1, 0]])

        elif degrees == 180:
            tf_rotate[0:2, 0:2] = np.array([[-1, 0], [0, -1]])

        elif degrees == 270:
            tf_rotate[0:2, 0:2] = np.array([[0, 1], [-1, 0]])

        elif degrees == 0:
            pass

        else:
            angle = values[0] * np.pi / 180.0
            tf_rotate[0:2, 0:2] = np.array([[np.cos(angle), -np.sin(angle)],
                                            [np.sin(angle), np.cos(angle)]])

        tf_offset_neg = np.identity(3)
        tf_offset_neg[0:2, 2:3] = np.array([[-offset[0]], [-offset[1]]])

        transform = tf_offset.dot(tf_rotate).dot(tf_offset_neg)

    elif 'skewX' in command:
        transform[0, 1] = np.tan(values[0] * np.pi / 180.0)

    elif 'skewY' in command:
        transform[1, 0] = np.tan(values[0] * np.pi / 180.0)

    else:
        assert False

    return transform


base_splitter = re.compile('(' + '|'.join(key for key in allowable) + ')')


replacement_compactifier_pairs = {
    re.compile(r'- [ ]*'): '-',
    re.compile(r'\+ [ ]*'): '+',
    re.compile(r', [ ]*'): ',',
    re.compile(r',\+'): ',',
    re.compile(r' \+'): ' ',
}


def turn_svg_transform_attribute_into_transform_tokens(string):
    assert isinstance(string, str)
    pieces = base_splitter.split(string)

    to_return = []
    last_command = None
    for piece in pieces:
        if piece == '':
            continue

        if piece in allowable:
            last_command = piece
            to_return.append(last_command)
            continue

        assert last_command is not None

        stripped = piece.strip().lstrip('(').rstrip(')').strip()
        assert '(' not in stripped
        assert ')' not in stripped

        compactified = stripped
        for regex, replacement in replacement_compactifier_pairs.items():
            compactified = regex.sub(replacement, compactified)

        string_tokens = list(float(t) for t in re.split('[ ,]', compactified) if t != '')
        number_tokens = [int_else_float(t) for t in string_tokens]

        assert len(number_tokens) in allowable[last_command]

        to_return.extend(number_tokens)
        last_command = None

    assert last_command is None
    return to_return


def is_a_times_b_matrix_as_nested_list(thing, a, b):
    assert isinstance(thing, list)

    if not len(thing) == a:
        return False

    for row in thing:
        if not isinstance(row, list) or len(row) != b:
            return False

        if not all(isinstance(u, Real) for u in row):
            return False

    return True


def is_2_by_3_matrix_as_nested_list(thing):
    return is_a_times_b_matrix_as_nested_list(thing, 2, 3)


def is_3_by_2_matrix_as_nested_list(thing):
    return is_a_times_b_matrix_as_nested_list(thing, 3, 2)


def turn_thing_into_transform_tokens(thing):
    if isinstance(thing, str):
        return [thing] if thing in allowable else turn_svg_transform_attribute_into_transform_tokens(thing)

    if isinstance(thing, Real):
        return [thing]

    if isinstance(thing, Complex):
        return [thing.real, thing.imag]

    if isinstance(thing, list):
        if is_3_by_2_matrix_as_nested_list(thing):
            return [thing[0][0], thing[0][1], thing[1][0], thing[1][1], thing[2][0], thing[2][1]]

        if is_2_by_3_matrix_as_nested_list(thing):
            return [thing[0][0], thing[1][0], thing[0][1], thing[1][1], thing[0][2], thing[1][2]]

        assert False

    try:
        x = thing.x 
        y = thing.y
        return [x, y]

    except AttributeError:
        pass

    print(thing)
    assert False


def string_and_values_packager(thing):
    cur_values = []

    for t in thing:
        if isinstance(t, str):
            if len(cur_values) > 0:
                yield cur_values
                cur_values = []
            yield t
            continue
        
        assert isinstance(t, Real)
        cur_values.append(t)

    if len(cur_values) > 0:
        yield cur_values


def turn_into_transform_tokens(*args):
    if len(args) == 1 and isinstance(args[0], list):
        thing = args[0]

        if is_2_by_3_matrix_as_nested_list(thing) or is_3_by_2_matrix_as_nested_list(thing):
            pass

        else:
            args = thing

    raw_tokens = []
    for a in args:
        raw_tokens.extend(turn_thing_into_transform_tokens(a))

    final_tokens = []
    for z in string_and_values_packager(raw_tokens):
        if isinstance(z, list):
            assert all(isinstance(u, Real) for u in z)
            if len(z) >= 6:
                i = len(z) % 6
                if i > 0:
                    assert len(final_tokens) > 0
                    assert isinstance(final_tokens[-1], str)
                    final_tokens.extend(z[:i])

                while i < len(z):
                    if len(final_tokens) == 0:
                        assert i == 0
                        final_tokens.append('matrix')

                    elif final_tokens[-1] != 'matrix':
                        assert isinstance(final_tokens[-1], Real)
                        final_tokens.append('matrix')

                    else:
                        assert i == 0
                    
                    final_tokens.extend(z[i:i + 6])
                    i += 6

                assert i == len(z)

            else:
                final_tokens.extend(z)

        elif isinstance(z, str):
            final_tokens.append(z)

        else:
            assert False

    return final_tokens


def string_and_values_iterator(tokens):
    command = None
    values = []

    def process_real_or_string_token(t):
        nonlocal command
        nonlocal values

        if isinstance(t, str):
            if command is not None:
                assert len(values) in allowable[command]
                yield command, values
                command = t
                values = []

            else:
                assert len(values) == 0
                command = t

        elif isinstance(t, Real):
            assert command is not None
            values.append(t)

        else:
            print("type(t)", type(t))
            print("t: ", t)
            assert False

    for t in tokens:
        if isinstance(t, list):
            for q in t:
                yield from process_real_or_string_token(q)
        
        else:
            yield from process_real_or_string_token(t)

    if command is not None:
        if len(values) not in allowable[command]:
            print(tokens)
        assert len(values) in allowable[command]
        yield command, values


def parse_transform(*args):
    if len(args) == 1 and isinstance(args[0], np.ndarray):
        assert is_svg_matrix(args[0])
        return args[0]

    tokens = turn_into_transform_tokens(*args)

    tf = np.identity(3)
    for (command, values) in string_and_values_iterator(tokens):
        tf = tf.dot(command_and_values_to_matrix(command, values))

    return tf


def generate_transform(*args, decimals=None, html_style=False):
    if len(args) == 1 and isinstance(args[0], np.ndarray):
        return matrix_to_string(args[0], decimals)

    def is_identity_transform_pair(command, values):
        return (
            (command == 'scale' and (values == [1] or values == [1, 1])) or
            (command == 'translate' and (values == [0] or values == [0, 0])) or
            (command == 'rotate' and values[0] == 0)
        )

    tokens = turn_into_transform_tokens(*args)

    return ' '.join(x for x in [
        command_and_values_to_string(command, values, decimals=decimals, html_style=html_style) for
        command, values in string_and_values_iterator(tokens) if not is_identity_transform_pair(command, values)
    ] if x != '')
