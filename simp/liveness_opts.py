from sympy import Symbol, Dummy

from pystencils import Field, Assignment

import random
import copy


def get_usage(atoms):
    reg_usage = {}
    for atom in atoms:
        reg_usage[atom.lhs] = 0
    for atom in atoms:
        for arg in atom.rhs.atoms():
            if isinstance(arg, Symbol) and not isinstance(arg, Field.Access):
                if arg in reg_usage:
                    reg_usage[arg] += 1
                else:
                    print(str(arg) + " is unsatisfied")
    return reg_usage


def get_definitions(eqs):
    definitions = {}
    for eq in eqs:
        definitions[eq.lhs] = eq
    return definitions


def get_roots(eqs):
    roots = []
    for eq in eqs:
        if isinstance(eq.lhs, Field.Access):
            roots.append(eq.lhs)
    if not roots:
        roots.append(eqs[-1].lhs)
    return roots


def merge_field_accesses(eqs):
    field_accesses = {}

    for eq in eqs:
        for arg in eq.rhs.atoms():
            if isinstance(arg, Field.Access) and arg not in field_accesses:
                field_accesses[arg] = Dummy()

    for i in range(0, len(eqs)):
        for f, s in field_accesses.items():
            if f in eqs[i].atoms():
                eqs[i] = eqs[i].subs(f, s)

    for f, s in field_accesses.items():
        eqs.insert(0, Assignment(s, f))

    return eqs


def refuse_eqs(input_eqs, max_depth=0, max_usage=1):
    eqs = copy.copy(input_eqs)
    usages = get_usage(eqs)
    definitions = get_definitions(eqs)

    def inline_trivially_schedulable(sym, depth):

        if sym not in usages or usages[sym] > max_usage or depth > max_depth:
            return sym

        rhs = definitions[sym].rhs
        if len(rhs.args) == 0:
            return rhs

        return rhs.func(*[inline_trivially_schedulable(arg, depth + 1) for arg in rhs.args])

    for idx, eq in enumerate(eqs):
        if usages[eq.lhs] > 1 or isinstance(eq.lhs, Field.Access):
            if not isinstance(eq.rhs, Symbol):

                eqs[idx] = Assignment(eq.lhs,
                                      eq.rhs.func(*[inline_trivially_schedulable(arg, 0) for arg in eq.rhs.args]))

    count = 0
    while (len(eqs) != count):
        count = len(eqs)
        usages = get_usage(eqs)
        eqs = [eq for eq in eqs if usages[eq.lhs] > 0 or isinstance(eq.lhs, Field.Access)]

    return eqs


def schedule_eqs(eqs, candidate_count=20):
    if candidate_count == 0:
        return eqs

    definitions = get_definitions(eqs)
    definition_atoms = {}
    for sym, definition in definitions.items():
        definition_atoms[sym] = list(definition.rhs.atoms(Symbol))
    roots = get_roots(eqs)
    initial_usages = get_usage(eqs)

    level = 0
    current_level_set = set([frozenset(roots)])
    current_usages = {frozenset(roots): {u: 0 for u in roots}}
    current_schedules = {frozenset(roots): (0, [])}
    max_regs = 0
    while len(current_level_set) > 0:
        new_usages = dict()
        new_schedules = dict()
        new_level_set = set()

        min_regs = min([len(current_usages[dec_set]) for dec_set in current_level_set])
        max_regs = max(max_regs, min_regs)
        candidates = [(dec_set, len(current_usages[dec_set])) for dec_set in current_level_set]

        random.shuffle(candidates)
        candidates.sort(key=lambda d: d[1])

        for dec_set, regs in candidates[:candidate_count]:
            for dec in dec_set:
                new_dec_set = set(dec_set)
                new_dec_set.remove(dec)
                usage = dict(current_usages[dec_set])
                usage.pop(dec)
                atoms = definition_atoms[dec]
                for arg in atoms:
                    if not isinstance(arg, Field.Access):
                        argu = usage.get(arg, initial_usages[arg]) - 1
                        if argu == 0:
                            new_dec_set.add(arg)
                        usage[arg] = argu
                frozen_new_dec_set = frozenset(new_dec_set)
                schedule = current_schedules[dec_set]
                max_reg_count = max(len(usage), schedule[0])

                if frozen_new_dec_set not in new_schedules or max_reg_count < new_schedules[frozen_new_dec_set][0]:

                    new_schedule = list(schedule[1])
                    new_schedule.append(definitions[dec])
                    new_schedules[frozen_new_dec_set] = (max_reg_count, new_schedule)

                if len(frozen_new_dec_set) > 0:
                    new_level_set.add(frozen_new_dec_set)
                new_usages[frozen_new_dec_set] = usage

        current_schedules = new_schedules
        current_usages = new_usages
        current_level_set = new_level_set

        level += 1

    schedule = current_schedules[frozenset()]
    schedule[1].reverse()
    return (schedule[1])


def liveness_opt_transformation(eqs):
    return refuse_eqs(merge_field_accesses(schedule_eqs(eqs, 3)), 1, 3)
