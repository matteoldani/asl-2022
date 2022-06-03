import networkx as nx
from matplotlib import colors
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import pygraphviz

fig, ax = plt.subplots(figsize=(15, 8))

relationships = pd.DataFrame({'from': ['bs1', 'bs1', 'bs2', 'alg opt1', 'alg opt1', 'alg opt1', 'alg opt2', 'opt0',
                                       'opt1', 'opt1', 'opt24', 'opt24', 'opt2', 'opt3', 'opt3', 'opt21', 'opt21',
                                       'opt3', 'opt31', 'opt3', 'opt3', 'opt3', 'opt33', 'opt35', 'opt42', 'opt43',
                                       'opt44', 'opt45', 'opt46', 'opt47', 'opt47', 'opt47', 'opt47', 'opt33', 'opt35',
                                       'opt51'],
                              'to': ['alg opt1', 'opt0', 'opt24', 'alg opt2', 'opt11', 'opt23', 'opt2', 'opt1', 'opt2',
                                     'opt24', 'opt12', 'opt41', 'opt3', 'opt11', 'opt21', 'opt22', 'opt23', 'opt31',
                                     'opt32', 'opt33', 'opt35', 'opt42', 'opt34', 'opt36', 'opt43', 'opt44', 'opt45',
                                     'opt46', 'opt47', 'opt37', 'opt48', 'opt51', 'opt53', 'opt51', 'opt53', 'opt54']})

# Create DF for node characteristics
carac = pd.DataFrame({'ID': ['bs1', 'bs2', 'alg opt1', 'alg opt2', 'opt0', 'opt1', 'opt2', 'opt3', 'opt11', 'opt21',
                             'opt31', 'opt32', 'opt33', 'opt34', 'opt35', 'opt36', 'opt42', 'opt24', 'opt12', 'opt41',
                             'opt23', 'opt22', 'opt43', 'opt44', 'opt45', 'opt46', 'opt47', 'opt37', 'opt48', 'opt53',
                             'opt51', 'opt54'],
                      'type': ['nor', 'bla', 'nor', 'nor', 'nor', 'nor', 'nor', 'nor', 'nor', 'nor', 'nor', 'nor',
                               'nor', 'nor', 'nor', 'nor', 'nor', 'bla', 'bla', 'bla', 'bla', 'bla', 'intri', 'intri',
                               'intri', 'intri', 'intri', 'intri', 'intri', 'intri', 'intri', 'intri']})

# Create graph object
G = nx.from_pandas_edgelist(relationships, 'from', 'to', create_using=nx.DiGraph())

# Make types into categories
carac = carac.set_index('ID')
carac = carac.reindex(G.nodes())

carac['type'] = pd.Categorical(carac['type'])
carac['type'].cat.codes

# Specify colors
cmap = colors.ListedColormap(['red', 'cyan', 'lightgrey'])

# Draw graph
nx.draw(
    G,
    with_labels=True,
    node_color=carac['type'].cat.codes,
    cmap=cmap, node_size=500,
    pos=graphviz_layout(G, prog="dot")
)
plt.savefig("dag.pdf", format="PDF")
