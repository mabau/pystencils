# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import json

from pystencils.backends.cbackend import CustomSympyPrinter, generate_c

try:
    import toml
except Exception:
    class toml:
        def dumps(self, *args):
            raise ImportError('toml not installed')

        def dump(self, *args):
            raise ImportError('toml not installed')

try:
    import yaml
except Exception:
    class yaml:
        def dumps(self, *args):
            raise ImportError('pyyaml not installed')

        def dump(self, *args):
            raise ImportError('pyyaml not installed')


def expr_to_dict(expr_or_node, with_c_code=True, full_class_names=False):

    self = {'str': str(expr_or_node)}
    if with_c_code:
        try:
            self.update({'c': generate_c(expr_or_node)})
        except Exception:
            try:
                self.update({'c': CustomSympyPrinter().doprint(expr_or_node)})
            except Exception:
                pass
    for a in expr_or_node.args:
        self.update({str(a.__class__ if full_class_names else a.__class__.__name__): expr_to_dict(a)})

    return self


def print_json(expr_or_node):
    dict = expr_to_dict(expr_or_node)
    return json.dumps(dict, indent=4)


def write_json(filename, expr_or_node):
    dict = expr_to_dict(expr_or_node)
    with open(filename, 'w') as f:
        json.dump(dict, f, indent=4)


def print_toml(expr_or_node):
    dict = expr_to_dict(expr_or_node, full_class_names=False)
    return toml.dumps(dict)


def write_toml(filename, expr_or_node):
    dict = expr_to_dict(expr_or_node)
    with open(filename, 'w') as f:
        toml.dump(dict, f)


def print_yaml(expr_or_node):
    dict = expr_to_dict(expr_or_node, full_class_names=False)
    return yaml.dump(dict)


def write_yaml(filename, expr_or_node):
    dict = expr_to_dict(expr_or_node)
    with open(filename, 'w') as f:
        yaml.dump(dict, f)
