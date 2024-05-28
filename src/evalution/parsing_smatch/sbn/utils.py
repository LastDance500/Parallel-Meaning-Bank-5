#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：PMB5.0.0 
@File ：utils.py
@Author ：xiao zhang
@Date ：2023/9/4 20:19 
'''

import re
'''
Various routines used by scores_nodes.py
'''
NEW_BOX_INDICATORS = {
    "ALTERNATION",
    "ATTRIBUTION",
    "CONDITION",
    "CONSEQUENCE",
    "CONTINUATION",
    "CONTRAST",
    "EXPLANATION",
    "NECESSITY",
    "NEGATION",
    "POSSIBILITY",
    "PRECONDITION",
    "RESULT",
    "SOURCE",
}

DRS_OPERATORS = {
    # Manually added (not part of clf_signature.yaml)
    "TSU",  # What does this mean?
    "MOR",
    "BOT",
    "TOP",
    "ESU",
    "EPR",
    # --- From here down copied from clf_signature.yaml ---
    # temporal relations
    "EQU",  # equal
    "NEQ",  # not equla
    "APX",  # approximate
    "LES",  # less
    "LEQ",  # less or equal
    "TPR",  # precede
    "TAB",  # abut
    "TIN",  # include
    # spatial operators
    "SZP",  # above x / y
    "SZN",  # under x \ y
    "SXP",  # behind x » y
    "SXN",  # before x « y
    "STI",  # inside
    "STO",  # outside
    "SY1",  # beside
    "SY2",  # between
    "SXY",  # around
}

INVERTIBLE_ROLES = {
    "InstanceOf",
    "AttributeOf",
    "ColourOf",
    "ContentOf",
    "PartOf",
    "SubOf",
}

ROLES = {
    # Manually added (not part of clf_signature.yaml)
    "InstanceOf",
    # --- From here down copied from clf_signature.yaml ---
    # Concept roles
    "Proposition",
    "Name",
    # Event roles
    "Agent",
    "Asset",
    "Attribute",
    "AttributeOf",
    "Beneficiary",
    "Causer",
    "Co-Agent",
    "Co-Patient",
    "Co-Theme",
    "Consumer",
    "Destination",
    "Duration",
    "Experiencer",
    "Finish",
    "Frequency",
    "Goal",
    "Instrument",
    "Instance",
    "Location",
    "Manner",
    "Material",
    "Path",
    "Patient",
    "Pivot",
    "Product",
    "Recipient",
    "Result",
    "Source",
    "Start",
    "Stimulus",
    "Theme",
    "Time",
    "Topic",
    "Value",
    # Concept roles
    "Bearer",
    "Colour",
    "ColourOf",
    "ContentOf",
    "Content",
    "Creator",
    "Degree",
    "MadeOf",
    # - Name
    "Of",
    "Operand",
    "Owner",
    "Part",
    "PartOf",
    "Player",
    "Quantity",
    "Role",
    "Sub",
    "SubOf",
    "Title",
    "Unit",
    "User",
    # Time roles
    "ClockTime",
    "DayOfMonth",
    "DayOfWeek",
    "Decade",
    "MonthOfYear",
    "YearOfCentury",
    # Other roles.
    "Affector",
    "Context",
    "Equal",
    "Extent",
    "Precondition",
    "Measure",
    "Cause",
    "Order",
    "Participant",
}

CONSTANTS = {
        "speaker",
        "hearer",
        "now",
        "unknown_ref",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }

MEMBERS = {
     "member"
    }

def disambig(lst):
    lst2 = []
    for v in lst:
        idx = 1
        v_idx = v + '_0'
        while str(v_idx) in lst2:
            v_idx = v + '_' + str(idx)
            idx += 1
        lst2.append(str(v_idx))
    return lst2

def concepts(v2c_dict):
    return [str(v) for v in v2c_dict.values() if re.search(r"(.+)\.(n|v|a|r)\.(\d+)", v)]

def con_noun(v2c_dict):
    return [str(v) for v in v2c_dict.values() if re.search(r"(.+)\.(n)\.(\d+)", v)]

def con_adj(v2c_dict):
    return [str(v) for v in v2c_dict.values() if re.search(r"(.+)\.(a)\.(\d+)", v)]

def con_adv(v2c_dict):
    return [str(v) for v in v2c_dict.values() if re.search(r"(.+)\.(r)\.(\d+)", v)]

def con_verb(v2c_dict):
    return [str(v) for v in v2c_dict.values() if re.search(r"(.+)\.(v)\.(\d+)", v)]

def namedent(v2c_dict, triples):
    return [str(v2c_dict[v1]) for (l,v1,v2) in triples if l == "Name"]

def negations(v2c_dict, triples):
    return [v2c_dict[v1] for (l,v1,v2) in triples if l == "NEGATION"]

def discources(v2c_dict, triples):
    return [v2c_dict[v1] for (l,v1,v2) in triples if l in NEW_BOX_INDICATORS and l != "NEGATION"]

def constants(v2c_dict):
    return[str(v) for v in v2c_dict.values() if v.strip('_') in CONSTANTS]

def roles(triples):
    return [str(l) for (l, v1, v2) in triples if l in ROLES]

def members(triples):
    return [str(l) for (l, v1, v2) in triples if l in MEMBERS]

def c2c(v2c_dict, triples): #concept2concpet
    lst = []
    vrs = []
    for t in triples:
        if t[0] in ROLES and "s" in t[1] and "s" in t[2]:## concepts is s0, s1
            #although the smatch code we use inverts the -of relations
            #there seems to be cases where this is not done so we invert
            #them here
            if t[0].endswith("Of"):
                lst.append((t[0][0:-3],t[2],t[1]))
                vrs.extend([t[2],t[1]])
            else:
                lst.append(t)
                vrs.extend([t[1],t[2]])

    #collect var/concept pairs for all extracted nodes
    dict1 = {}
    for i in v2c_dict:
        if i in vrs:
            dict1[i] = v2c_dict[i]
    return (lst, dict1)

def c2n(v2c_dict, triples):
    lst = []
    vrs = []
    for t in triples:
        if t[0] == 'Name':
            lst.append(t)
            vrs.extend([t[1],t[2]])
    dict1 = {}
    for i in v2c_dict:
        if i in vrs:
            dict1[i] = v2c_dict[i]
    return (lst, dict1)

def b2c(v2c_dict, triples): # box member concept
    lst = []
    vrs = []
    for t in triples:
        if t[0] == "member":
            lst.append(t)
            vrs.extend([t[1],t[2]])
    #collect var/concept pairs for all extracted nodes
    dict1 = {}
    for i in v2c_dict:
        if i in vrs:
            dict1[i] = v2c_dict[i]
    return (lst, dict1)

def c2o(v2c_dict, triples):
    lst = []
    vrs = []
    for t in triples:
        if t[0] in DRS_OPERATORS and t[0] != "TOP":
            lst.append(t)
            vrs.extend([t[1],t[2]])
    #collect var/concept pairs for all extracted nodes
    dict1 = {}
    for i in v2c_dict:
        if i in vrs:
            dict1[i] = v2c_dict[i]
    return (lst, dict1)


def b2b(v2c_dict, triples): # box CONSTRAST box without negation
    lst = []
    vrs = []
    for t in triples:
        if t[0] in NEW_BOX_INDICATORS and t[0] != "TOP" and t[0] != "NEGATION":
            lst.append(t)
            vrs.extend([t[1],t[2]])
    #collect var/concept pairs for all extracted nodes
    dict1 = {}
    for i in v2c_dict:
        if i in vrs:
            dict1[i] = v2c_dict[i]
    return (lst, dict1)

def var2concept(amr):
    v2c = {}
    for n, v in zip(amr.nodes, amr.node_values):
        v2c[n] = v
    return v2c