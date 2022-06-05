import networkx as nx
from matplotlib import colors
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd

fig, ax = plt.subplots(figsize=(15, 8))

relationships = pd.DataFrame({'from': ['bs1', 'bs1', 'bs2', 'alg opt1', 'alg opt1', 'alg opt1', 'alg opt2', 'opt0',
                                       'opt1', 'opt1', 'opt24', 'opt24', 'opt2', 'opt3', 'opt3', 'opt21', 'opt21',
                                       'opt3', 'opt31', 'opt3', 'opt3', 'opt3', 'opt33', 'opt35', 'opt42', 'opt43',
                                       'opt44', 'opt45', 'opt46', 'opt47', 'opt47', 'opt47', 'opt47', 'opt33', 'opt35',
                                       'opt51', 'bs2', 'opt47', 'opt53'],
                              'to': ['alg opt1', 'opt0', 'opt24', 'alg opt2', 'opt11', 'opt23', 'opt2', 'opt1', 'opt2',
                                     'opt24', 'opt12', 'opt41', 'opt3', 'opt11', 'opt21', 'opt22', 'opt23', 'opt31',
                                     'opt32', 'opt33', 'opt35', 'opt42', 'opt34', 'opt36', 'opt43', 'opt44', 'opt45',
                                     'opt46', 'opt47', 'opt37', 'opt48', 'opt51', 'opt53', 'opt51', 'opt53', 'opt54',
                                     'opt23', 'opt60', 'opt61']})

# Create DF for node characteristics
carac = pd.DataFrame({'ID': ['bs1', 'bs2', 'alg opt1', 'alg opt2', 'opt0', 'opt1', 'opt2', 'opt3', 'opt11', 'opt21',
                             'opt31', 'opt32', 'opt33', 'opt34', 'opt35', 'opt36', 'opt42', 'opt24', 'opt12', 'opt41',
                             'opt23', 'opt22', 'opt43', 'opt44', 'opt45', 'opt46', 'opt47', 'opt37', 'opt48', 'opt53',
                             'opt51', 'opt54', 'opt60', 'opt61'],
                      'type': ['lightslategrey', 'mediumblue', 'lightslategrey', 'lightslategrey', 'lightslategrey',
                               'lightslategrey',
                               'lightslategrey', 'lightslategrey', 'lightslategrey', 'lightslategrey', 'lightslategrey',
                               'lightslategrey',
                               'lightslategrey', 'lightslategrey', 'lightslategrey', 'lightslategrey', 'lightslategrey',
                               'mediumblue',
                               'mediumblue', 'mediumblue', 'mediumblue', 'mediumblue', 'firebrick', 'firebrick',
                               'firebrick', 'firebrick', 'firebrick', 'firebrick', 'firebrick',
                               'firebrick', 'firebrick', 'firebrick', 'firebrick', 'firebrick']})
# mediumblue,
# Create graph object
G = nx.from_pandas_edgelist(relationships, 'from', 'to', create_using=nx.DiGraph())

# Make types into categories
# carac = carac.set_index('ID')
# carac = carac.reindex(G.nodes())

carac['type'] = pd.Categorical(carac['type'])

A = nx.nx_agraph.to_agraph(G)

A.layout(prog='dot')
A.node_attr['style'] = 'filled'

for index, row in carac.iterrows():
    n = A.get_node(row['ID'])
    n.attr['fontcolor'] = 'whitesmoke'
    n.attr["color"] = row['type']

A.draw('dag.pdf', args='-Gnodesep=0.01 -Gfont_size=1 -Grankdir="LR"', prog='dot')
