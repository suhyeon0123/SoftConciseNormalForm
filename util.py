from FAdo.fa import *
from FAdo.cfg import *
from parsetree import*

import copy

def membership(regex, string):
    return bool(re.fullmatch(regex, string))


def is_solution(regex, examples, membership):
    if regex == '@emptyset':
        return False

    for string in examples.getPos():
        if not membership(regex, string):
            return False

    for string in examples.getNeg():
        if membership(regex, string):
            return False

    return True


def is_pdead(s, examples):
    s_spreadAll = s.spreadAll()

    for string in examples.getPos():
        if not membership(s_spreadAll, string):
            return True
    return False


def is_ndead(s, examples):
    s_spreadNP = s.spreadNP()

    if s_spreadNP == '@emptyset':
        return False

    for string in examples.getNeg():
        if membership(s_spreadNP, string):
            return True

    return False


def is_not_scnf(s, new_elem):
    if repr(new_elem) == '0|1' or new_elem.type == Type.CHAR:
        checker = True
    else:
        checker = False

    if (new_elem.type == Type.K or new_elem.type == Type.Q) and s.starnormalform():
        # print(repr(k), "starNormalForm")
        return True

    if checker and s.redundant_concat1():
        # print("concat1")
        return True

    if s.redundant_concat2():
        # print("concat2")
        return True

    if checker and s.KCK():
        # print(repr(k), "is kc_qc")
        return True

    if (new_elem.type == Type.K or new_elem.type == Type.Q or checker) and s.KCQ():
        # print(repr(k), "KCQ")
        return True

    if checker and s.QC():
        # print(repr(k), "is kc_qc")
        return True

    if new_elem.type == Type.Q and s.OQ():
        # print(repr(k), "is OQ")
        return True

    if checker and s.orinclusive():
        # print(repr(k), "is orinclusive")
        return True

    if checker and s.prefix():
        # print(repr(k), "is prefix")
        return True

    if (new_elem.type == Type.K or new_elem.type == Type.Q or checker) and s.sigmastar():
        # print(repr(k), "is equivalent_KO")
        return True
    return False


def is_redundant(s, examples, new_elem):
    if repr(new_elem) == '0|1' or new_elem.type == Type.CHAR:
        checker = True
    else:
        checker = False
    if not (new_elem.type == Type.Q or checker):
        return False

    # unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.prior_unroll()
    tmp = unrolled_state.reprAlpha2()
    unsp = list(i.replace('#', '(0|1)*') for _, i in tmp)

    # check part
    for state in unsp:
        is_red = True
        for string in examples.getPos():
            if membership(state, string):
                is_red = False
                break
        if is_red:
            return True
    return False



