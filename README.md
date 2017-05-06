
Welcome to Pykov docs
====================

Pykov is a tiny Python module on *finite regular Markov chains*.

You can define a Markov chain from scratch or read it from a text file according specific format. Pykov is versatile, being it able to manipulate the chain, inserting and removing nodes, and to calculate various kind of quantities, like the steady state distribution, mean first passage times, random walks, absorbing times, and so on.

Pykov is licensed under the terms of the **GNU General Public License** as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

---------------

Installation
-------------
Pykov can be installed/upgraded via [`pip`<i class="icon-forward"></i>](http://pip.readthedocs.org/en/latest/#) 
```sh
$ pip install git+git://github.com/riccardoscalco/Pykov@master #both Python2 and Python3
$ pip install --upgrade git+git://github.com/riccardoscalco/Pykov@master
```
Note that Pykov depends on **numpy** and **scipy**.

-----------------

Getting started
------------------

Open your favourite Python shell and import pykov:
```python
>>> import pykov
>>> pykov.__version__
```

###Vector class
The **Vector class** inherits from python `collections.OrderedDict`, which means it has the same behaviors and features of python ordered dictionaries, with few exceptions. The states and the corresponding probabilities are the keys and the values of the dictionary, respectively.

collections.OrderedDict is used instead of the default dict because from Python 3.3 onwards, dict is non-deterministic for security reasons (see this [stackoverflow question](http://stackoverflow.com/questions/14956313/dictionary-ordering-non-deterministic-in-python3)). For programs that needs determinism (such as simulations), an OrderedDict should be passed. But for programs where determinism is not an issue, dict can be used.

Definition of a `pykov.Vector()`:
```python
>>> p = pykov.Vector()
```
You can *get* and *set* states in many ways:
```python
>>> p['A'] = .2
>>> p
{'A': 0.2}

>>> # Non-deterministic example
>>> p = pykov.Vector({'A':.3, 'B':.7})
>>> p
{'A':0.3, 'B':0.7}

>>> # Deterministic example
>>> data = collections.OrderedDict((('A', .3), ('B', .7)))
>>> p = pykov.Vector(data)
>>> p
{'A':0.3, 'B':0.7}

>>> pykov.Vector(A=.3, B=.6, C=.1)
{'A':0.3, 'B':0.6, 'C':0.1}
```
States not belonging to the vector have zero probability, moreover states with zero probability are not shown:
```python
>>> q = pykov.Vector(C=.4, B=.6)
>>> q['C']
0.4
>>> q['Z']
0.0
>>> 'Z' in q
False

>>> q['Z']=.2
>>> q
{'C': 0.4, 'B': 0.6, 'Z': 0.2}
>>> 'Z' in q
True
>>> q['Z']=0
>>> q
{'C': 0.4, 'B': 0.6}
>>> 'Z' in q
False
```

#### Vector operations
##### **Sum**
A `pykov.Vector()` instance can be added or substracted to another `pykov.Vector()` instance:
```python
>>> p = pykov.Vector(A=.3, B=.7)
>>> q = pykov.Vector(C=.5, B=.5)
>>> p + q
{'A': 0.3, 'C': 0.5, 'B': 1.2}
>>> p - q
{'A': 0.3, 'C': -0.5, 'B': 0.2}
>>> q - p
{'A': -0.3, 'C': 0.5, 'B': -0.2}
```

##### **Product**
A `pykov.Vector()` instance can be multiplied by a scalar. The *dot product* with another `pykov.Vector()` or `pykov.Matrix()` instance is also supported.
```python
>>> p = pykov.Vector(A=.3, B=.7)
>>> p * 3
{'A': 0.9, 'B': 2.1}
>>> 3 * p
{'A': 0.9, 'B': 2.1}
>>> q = pykov.Vector(C=.5, B=.5)
>>> p * q
0.35
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> p * T
{'A': 0.91, 'B': 0.09}
>>> T * p
{'A': 0.42, 'B': 0.3}
```

#### Vector methods

##### **sum()**
Sum the probability values.
```python
>>> p = pykov.Vector(A=.3, B=.7)
>>> p.sum()
1.0
```

##### **sort(reverse=False)**
Return a list of tuples `(state, probability)` sorted according the probability values.
```python
>>> p = pykov.Vector({'A':.3, 'B':.1, 'C':.6})
>>> p.sort()
[('B', 0.1), ('A', 0.3), ('C', 0.6)]
>>> p.sort(reverse=True)
[('C', 0.6), ('A', 0.3), ('B', 0.1)]
```
##### **choose(random_func=None)**
Choose a state at random, according to its probability.
```python
>>> p = pykov.Vector(A=.3, B=.7)
>>> p.choose()
'B'
>>> p.choose()
'B'
>>> p.choose()
'A'
```

Optionally, if you need to supply your own random number generator, you can pass a function that takes two inputs (the min and max) and Pykov will use that function.
```python
>>> def FakeRandom(min, max): return 0.01
>>> p = pykov.Vector(A=.05, B=.4, C=.4, D=.15)
>>> p.choose(FakeRandom)
'A'        
```


##### **normalize()**
Normalize the `pykov.Vector`, after normalization the probabilities sum to 1.
```python
>>> p = pykov.Vector({'A':3, 'B':1, 'C':6})
>>> p.sum()
10.0
>>> p.normalize()
>>> p
{'A': 0.3, 'C': 0.6, 'B': 0.1}
>>> p.sum()
1.0
```

##### **copy()**
Return a shallow copy.
```python
>>> p = pykov.Vector(A=.3, B=.7)
>>> q = p.copy()
>>> p['C'] = 1.
>>> q
{'A': 0.3, 'B': 0.7}
```

##### **entropy()**
Return the Shannon entropy, defined as $H(p) = \sum_i p_i \ln p_i$.
```python
>>> p = pykov.Vector(A=.3, B=.7)
>>> p.entropy()
0.6108643020548935
```
For further details, have a look at *Khinchin A. I., Mathematical Foundations of Information Theory Dover, 1957*.

##### **dist(q)**
Return the distance to another `pykov.Vector`, defined as $d(p,q) = \sum_i | p_i - q_i |$. 
```python
>>> p = pykov.Vector(A=.3, B=.7)
>>> q = pykov.Vector(C=.5, B=.5)
>>> p.dist(q)
1.0
```

##### **relative_entropy(q)**
Return the *Kullback-Leibler* distance, defined as $d(p,q) = \sum_i p_i \ln (p_i/q_i)$.
```python
>>> p = pykov.Vector(A=.3, B=.7)
>>> q = pykov.Vector(A=.4, B=.6)
>>> p.relative_entropy(q) #d(p,q)
0.02160085414354654
>>> q.relative_entropy(p) #d(q,p)
0.022582421084357485
```
Note that the Kullback-Leibler distance is not symmetric.

------------

### Matrix class
The `pykov.Matrix()` class inherits from python collections.OrderedDict. Similar to the default dict, OrderedDict `keys` are `tuple` of states, OrderedDict `values` are the matrix entries. Indexes do not need to be `int`, they can be `string`, as the states of a `pykov.Vector()`.

Definition of  `pykov.Matrix()`:
```python
>>> T = pykov.Matrix()
```
You can *get* and *set* items in many ways:
```python
>>> T = pykov.Matrix()
>>> T[('A','B')] = .3
>>> T
{('A', 'B'): 0.3}
>>> T['A','A'] = .7
>>> T
{('A', 'B'): 0.3, ('A', 'A'): 0.7}

>>> # Non-deterministic example
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T[('A','B')]
0.3
>>> T['A','B']
0.3

>>> # deterministic example
>>> data = collections.OrderedDict((
                (('A','B'), .3), 
                (('A','A'), .7), 
                (('B','A'), 1.)))
>>> T = pykov.Matrix(data)
>>> T[('A','B')]
0.3
>>> T['A','B']
0.3
```

Items not belonging to the matrix have value equal to zero, moreover items with value equal to zero are not shown:
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T['B','B']
0.0

>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T
{('A', 'B'): 0.3, ('A', 'A'): 0.7, ('B', 'A'): 1.0}
>>> T['A','A'] = 0
>>> T
{('A', 'B'): 0.3, ('B', 'A'): 1.0}
```

#### Matrix operations

##### **Sum**
A `pykov.Matrix()` instance can be added or substracted to another `pykov.Matrix()` instance.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> I = pykov.Matrix({('A','A'):1, ('B','B'):1})
>>> T + I
{('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 1.7, ('B', 'B'): 1.0}
>>> T - I
{('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): -0.3, ('B', 'B'): -1}
```

##### **Product**
A `pykov.Matrix()` instance can be multiplied by a scalar, the dot product with a `pykov.Vector()` or another `pykov.Matrix()` instance is also supported.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T * 3
{('B', 'A'): 3.0, ('A', 'B'): 0.9, ('A', 'A'): 2.1}

>>> p = pykov.Vector(A=.3, B=.7)
>>> T * p
{'A': 0.42, 'B': 0.3}

>>> W = pykov.Matrix({('N', 'M'): 0.5, ('M', 'N'): 0.7,
                      ('M', 'M'): 0.3, ('O', 'N'): 0.5,
                      ('O', 'O'): 0.5, ('N', 'O'): 0.5})
>>> W * W
{('N', 'M'): 0.15, ('M', 'N'): 0.21, ('M', 'O'): 0.35,
 ('M', 'M'): 0.44, ('O', 'M'): 0.25, ('O', 'N'): 0.25,
 ('O', 'O'): 0.5, ('N', 'O'): 0.25, ('N', 'N'): 0.6}
```

#### Matrix methods

##### **states()**
Return the `set` of states.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.states()
{'B', 'A'}
```
States without ingoing or outgoing transitions are removed from the set of states.
```python
>>> T['A','C']=1
>>> T.states()
{'A', 'C', 'B'}
>>> T['A','C']=0
>>> T.states()
{'A', 'B'}
```

##### **pred(key=None)**
Return the precedessors of a state (if not indicated, of all states). In matrix notation, return the column of the indicated state.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.pred()
{'A': {'A': 0.7, 'B': 1.0}, 'B': {'A': 0.3}}
>>> T.pred('A')
{'A': 0.7, 'B': 1.0}
```

##### **succ(key=None)**
Return the successors of a state (if not indicated, of all states). In matrix notation, return the row of the indicated state.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.succ()
{'A': {'A': 0.7, 'B': 0.3}, 'B': {'A': 1.0}}
>>> T.succ('A')
{'A': 0.7, 'B': 0.3}
```

##### **copy()**
Return a shallow copy.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> W = T.copy()
>>> T[('B','B')] = 1.
>>> W
{('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 0.7}
```

##### **remove(states)**
Return a shallow copy of the matrix without the indicated states.
All the links where the states appear are deleted, so that the result will not be in general a stochastic matrix.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.remove(['B'])
{('A', 'A'): 0.7}
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.,
                     ('C','D'): .5, ('D','C'): 1., ('C','B'): .5})
>>> T.remove(['A','B'])
{('C', 'D'): 0.5, ('D', 'C'): 1.0}
```

##### **stochastic()**
Change the `pykov.Matrix()` instance in a right [stochastic matrix](http://en.wikipedia.org/wiki/Stochastic_matrix).
Set the sum of every row equal to one, raise `PykovError` if not possible.
```python
>>> T = pykov.Matrix({('A','B'): 3, ('A','A'): 7, ('B','A'): .2})
>>> T.stochastic()
>>> T
{('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 0.7}

>>> T[('A','C')]=1
>>> T.stochastic()
pykov.PykovError: 'Zero links from node C'
```

##### **transpose()**
Return the transpose of the `pykov.Matrix()` instance.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.transpose()
{('B', 'A'): 0.3, ('A', 'B'): 1.0, ('A', 'A'): 0.7}
>>> T
{('A', 'B'): 0.3, ('A', 'A'): 0.7, ('B', 'A'): 1.0}
```

##### **eye()**
Return the [Identity matrix](http://en.wikipedia.org/wiki/Identity_matrix).
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.eye()
{('A', 'A'): 1., ('B', 'B'): 1.}
>>> type(T.eye())
<class 'pykov.Matrix'>
```

##### **ones()**
Return a `pykov.Vector()` instance with entries equal to 1.
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.ones()
{'A': 1.0, 'B': 1.0}
>>> type(T.ones())
<class 'pykov.Vector'>
```

##### **trace()**
Return the matrix [trace](http://en.wikipedia.org/wiki/Trace_%28linear_algebra%29).
```python
>>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.trace()
0.7
```

---------------

### Chain class

The `pykov.Chain` class inherits from `pykov.Matrix` class.
The OrderedDict `key` is a tuple of states, the OrderedDict `value` is the transition
probability to go from the first state to the second state, in other words
pykov describes the transitions of a Markov chain with a *right* stochastic matrix.

#### Chain methods

##### **adjacency()**
Return the [adjacency matrix](http://en.wikipedia.org/wiki/Adjacency_matrix).
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.adjacency()
{('B', 'A'): 1, ('A', 'B'): 1, ('A', 'A'): 1}
>>> type(T.adjacency())
<class 'pykov.Matrix'>
```

##### **pow(p, n)**
Find the probability distribution after `n` steps, starting from an initial `pykov.Vector()` `p`.
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> p = pykov.Vector(A=1)
>>> T.pow(p,3)
{'A': 0.7629999999999999, 'B': 0.23699999999999996}
>>> p * T * T * T #not efficient
{'A': 0.7629999999999999, 'B': 0.23699999999999996}
```

##### **move(state)**
Do one step from the indicated `state` to one of its successors, chosen at random according to the transition probability. Return the new state.
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.move('A')
'B'
```

Optionally, if you need to supply your own random number generator, you can pass a function that takes two inputs (the min and max) and Pykov will use that function.
```python
>>> def FakeRandom(min, max): return 0.01
>>> T.move('A', FakeRandom)
'B'        
```

##### **walk(steps, start=None, stop=None)**
Return a random walk of n `steps`, starting and stopping at the indicated states.  If not indicated, then the starting state is chosen according to the steady state probability. If the stopping state is reached before to do n steps, then the walker stops.
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.walk(10)
['B', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A']
>>> T.walk(10,'B','B')
['B', 'A', 'A', 'A', 'A', 'A', 'B']
```

##### **walk_probability(walk)**
Return the *logarithm* of the **walk** probability (see `walk()` method). Impossible walks have probability equal to zero.
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.walk_probability(['A','A','B','A','A'])
-1.917322692203401
>>> probability = math.exp(-1.917322692203401)
>>> probability
0.147

>>> p = T.walk_probability(['A','B','B','B','A'])
>>> math.exp(p)
0.0
```


##### **steady()**
Return the steady state, i.e. the equilibrium distribution of the chain.
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.steady()
{'A': 0.7692307692307676, 'B': 0.23076923076923028}
```
Since Pykov describes the chain with a right stochatic matrix,
the steady state $x$ satisfies at the condition $p=pT$
and it is calculated with the *inverse iteration method* 
$Q^t x = e$, where $Q = I - T$ and $e = (0,0,...,1)$.
Moreover, the Markov chain is assumed to be ergodic, i.e. the transition matrix
must be irreducible and acyclic.
You can easily test such properties by means of
[NetworkX](http://networkx.github.io/), let's see how:
```python
>>> import networkx as nx

>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> G = nx.DiGraph(list(T.keys()))
>>> nx.is_strongly_connected(G) # is irreducible
True
>>> nx.is_aperiodic(G)
True

>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','B'): 1.})
>>> G = nx.DiGraph(list(T.keys()))
>>> nx.is_strongly_connected(G) # is irreducible
False
>>> nx.is_aperiodic(G)
True

>>> T = pykov.Chain({('A','B'): 1, ('B','C'): 1., ('C','A'): 1.})
>>> G = nx.DiGraph(list(T.keys()))
>>> nx.is_strongly_connected(G)
True
>>> nx.is_aperiodic(G)
False
```

Often, Markov chains created from raw data are not irreducibles.
In such cases, the Markov chain may be defined by means of the
largest strongly connected component of the associated graph.
Strongly connected components can be found with NetworkX:
```python
>>> nx.strongly_connected_components(G)
```

For further details on the inverse iteration method,
have a look at *W. Stewart, Introduction to the Numerical Solution of
Markov Chains, Princeton University Press, Chichester, West Sussex, 1994*.

##### **mixing_time(cutoff=0.25, jump=1, p=None)**
Return the [mixing time](http://en.wikipedia.org/wiki/Markov_chain_mixing_time), defined as the number of steps needed to have $|pT^n - \pi| \lt 0.25$, where $\pi$ is the steady state $\pi = \pi T$.

If the initial distribution `p` is not indicated, then the iteration starts from the less probable state of the steady distribution. The parameter `jump` controls the iteration step, for example with `jump=2` n has values 2,4,6,8,..
```python
>>> d = {('R','R'):1./2, ('R','N'):1./4, ('R','S'):1./4,
         ('N','R'):1./2, ('N','N'):0., ('N','S'):1./2,
         ('S','R'):1./4, ('S','N'):1./4, ('S','S'):1./2}
>>> T = pykov.Chain(d)
>>> T.mixing_time()
2
```

##### **entropy(p=None, norm=False)**
Return the Chain entropy, defined as $H = \sum_i \pi_i H_i$, where $H_i=\sum_j T_{ij}\ln T_{ij}$.
If `p` is not `None`, then the entropy is calculated with the indicated probability `pykov.Vector()`.
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.entropy()
0.46989561696530169
```
With **norm=True** entropy belongs to [0,1].
```python
>>> T.entropy(norm=True)
0.33895603665233132
```
For further details, have a look at *Khinchin A. I., Mathematical Foundations of Information Theory, Dover, 1957*.

##### **mfpt_to(state)**
Return the Mean First Passage Times of every state *to* the indicated `state`.
```python
>>> d = {('R', 'N'): 0.25, ('R', 'S'): 0.25, ('S', 'R'): 0.25,
         ('R', 'R'): 0.5, ('N', 'S'): 0.5, ('S', 'S'): 0.5,
         ('S', 'N'): 0.25, ('N', 'R'): 0.5, ('N', 'N'): 0.0}
>>> T = pykov.Chain(d)
>>> T.mfpt_to('R')
{'S': 3.333333333333333, 'N': 2.666666666666667} #mfpt from 'S' to 'R' is 3.33
```
See also *Kemeny J. G. and Snell, J. L., Finite Markov Chains. Springer-Verlag: New York, 1976*.

##### **absorbing_time(transient_set)**
Mean number of steps needed to leave the `transient_set`, return the `pykov.Vector()` `tau` where `tau[i]` is the mean number of steps needed to leave the transient set starting from state `i`. The parameter `transient_set` is a subset of states (iterable).
```python
>>> d = {('R','R'):1./2, ('R','N'):1./4, ('R','S'):1./4,
         ('N','R'):1./2, ('N','N'):0., ('N','S'):1./2,
         ('S','R'):1./4, ('S','N'):1./4, ('S','S'):1./2}
>>> T = pykov.Chain(d)
>>> p = pykov.Vector({'N':.3, 'S':.7})
>>> tau = T.absorbing_time(p.keys())
>>> tau
{'S': 3.333333333333333, 'N': 2.6666666666666665}
```
In other words, the mean number of steps in order to leave states `'S'` and `'N'` starting from `'S'` is 3.33.
It is sufficient to calculate `p * tau` in order to weight the mean times according an initial distribution `p`.
```python
>>> p * tau
3.1333333333333329
```
In order to better understand the meaning of the method, the calculation of the previous example can be approximated by means of many random walkers:
```python
>>> numpy.mean([len(T.walk(10000000,"S","R"))-1 for i in range(1000000)])
3.3326020000000001
>>> numpy.mean([len(T.walk(10000000,"N","R"))-1 for i in range(1000000)])
2.6665549999999998
```

##### **absorbing_tour(p, transient_set=None)**
Return a `pykov.Vector()` `v`, where `v[i]` is the mean time the process is in the transient state `i` before leaving the transient set.

Note that `v.sum()` is equal to `p * tau` (see `absorbing_time()` method).
If not specified, the transient set is defined as the set of states in vector `p`.
```python
>>> d = {('R','R'):1./2, ('R','N'):1./4, ('R','S'):1./4,
         ('N','R'):1./2, ('N','N'):0., ('N','S'):1./2,
         ('S','R'):1./4, ('S','N'):1./4, ('S','S'):1./2}
>>> T = pykov.Chain(d)
>>> p = pykov.Vector({'N':.3, 'S':.7})
>>> T.absorbing_tour(p)
{'S': 2.2666666666666666, 'N': 0.8666666666666669}
```

##### **fundamental_matrix()**
Return the fundamental matrix, have a look at *Kemeny J. G. and Snell J. L., Finite Markov Chains. Springer-Verlag: New York, 1976* for further details.
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.fundamental_matrix()
{('B', 'A'): 0.17751479289940991, ('A', 'B'): 0.053254437869822958,
('A', 'A'): 0.94674556213017902, ('B', 'B'): 0.82248520710059214}
```
Note that the fundamental matrix is not sparse.

##### **kemeny_constant()**
Return the Kemeny constant of the transition matrix.
```python
>>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
>>> T.kemeny_constant()
1.7692307692307712
```

----------------------------

Utilities
----------
Pykov comes with an utility useful to create a `pykov.Chain()` from a text file, let say file `/mypath/mat`, which contains the transition matrix defined with the following format:
```python
A A .7
A B .3
B A 1
```
The `pykov.Chain()` instance is created with the command:
```python
>>> P = pykov.readmat('/mypath/mat')
>>> P
{('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 0.7}
```
-----------------------------------
> Docs written with [StackEdit](https://stackedit.io/).
