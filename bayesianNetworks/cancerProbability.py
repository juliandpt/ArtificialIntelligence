# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:56:01 2019

@author: Julian de Pablo
"""

from pomegranate import *

asia = DiscreteDistribution({'V': 0.01, 'F': 0.99})
fumador = DiscreteDistribution({'V': 0.5, 'F': 0.5})
tuberculosis = ConditionalProbabilityTable(
        [['V', 'V', 0.2],
         ['V', 'F', 0.8],
         ['F', 'V', 0.01],
         ['F', 'F',0.99]], [asia])

cancer = ConditionalProbabilityTable(
        [['V', 'V', 0.75],
         ['V', 'F', 0.25],
         ['F', 'V', 0.02],
         ['F', 'F',0.98]], [fumador])

bronquitis = ConditionalProbabilityTable(
        [['V', 'V', 0.92],
         ['V', 'F', 0.08],
         ['F', 'V', 0.03],
         ['F', 'F',0.97]], [fumador])

ToC = ConditionalProbabilityTable(
        [['V', 'V', 'V', 1],
         ['V', 'F', 'V', 1],
         ['F', 'V', 'V', 1],
         ['F', 'F', 'V', 0],
         ['V', 'V', 'F', 0],
         ['V', 'F', 'F', 0],
         ['F', 'V', 'F', 0],
         ['F', 'F', 'F', 1]],[tuberculosis, cancer])

disnea = ConditionalProbabilityTable(
        [['V', 'V', 'V', 0.96],
         ['V', 'F', 'V', 0.89],
         ['F', 'V', 'V', 0.96],
         ['F', 'F', 'V', 0.89],
         ['V', 'V', 'F', 0.04],
         ['V', 'F', 'F', 0.11],
         ['F', 'V', 'F', 0.04],
         ['F', 'F', 'F', 0.11]],[ToC, bronquitis])

rayosX = ConditionalProbabilityTable(
        [['V', 'V', 0.885],
         ['V', 'F', 0.115],
         ['F', 'V', 0.04],
         ['F', 'F',0.96]], [ToC])

s1 = Node(asia, name="asia")
s2 = Node(fumador, name="fumador")
s3 = Node(tuberculosis, name="tuberculosis")
s4 = Node(cancer, name="cancer")
s5 = Node(bronquitis, name="bronquitis")
s6 = Node(ToC, name="ToC")
s7 = Node(disnea, name="disnea")
s8 = Node(rayosX, name="rayosX")


network = BayesianNetwork("Cancer Problem")
network.add_nodes(s1, s2, s3, s4, s5, s6, s7, s8)
network.add_edge(s1, s3)

network.add_edge(s2, s4)
network.add_edge(s2, s5)

network.add_edge(s3, s6)
network.add_edge(s4, s6)

network.add_edge(s6, s7)
network.add_edge(s5, s7)

network.add_edge(s6, s8)

network.bake()
 
observations1 = { 'tuberculosis' : 'F', 'fumador' : 'V', 'disnea' : 'V', 'rayosX' : 'V' }
observations2 = { 'tuberculosis' : 'F', 'fumador' : 'V', 'bronquitis' : 'V' }
beliefs = map(str, network.predict_proba(observations2))
print ("\n".join( "{}\t\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))