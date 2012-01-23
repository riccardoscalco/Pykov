# -*- coding: utf-8 -*-

# PyKov is Python package for the creation, manipulation and study of Markov
# Chains.
# Copyright (C) 2011  Riccardo Scalco
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Email: riccardo.scalco@gmail.com

"""Pykov documentation.

.. module:: A Python module for finite Markov chains.
   :platform: Unix, Windows, Mac

.. moduleauthor::
   Riccardo Scalco <riccardo.scalco@gmail.com>

"""


import random
import math

import numpy
import pysparse
#import networkx

__date__ = 'Jan 2012'

__version__ = 1.0

__license__ = 'GNU General Public License Version 3' 

__authors__ = 'Riccardo Scalco'

__many_thanks_to__ = 'Sandra Steiner'

def _del_cache(fn):
    """
    Delete cache.
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        #try:
        #    del self.__dict__["states"]
        #except KeyError:
        #    pass
        try:
            del(self._states)
        except AttributeError:
            pass
        try:
            del(self._succ)
        except AttributeError:
            pass
        try:
            del(self._pred)
        except AttributeError:
            pass
        try:
            del(self._steady)
        except AttributeError:
            pass
        try:
            del(self._guess)
        except AttributeError:
            pass
        try:
            del(self._fundamental_matrix)
        except AttributeError:
            pass
        return fn(*args, **kwargs)
    return wrapper

class PykovError(Exception):
    """
    Exception definition form Pykov Errors.
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Vector(dict):
    """
    """

    def __init__(self, data=None, **kwargs):
        """
        >>> pykov.Vector({'A':.3, 'B':.7})
        {'A':.3, 'B':.7}
        >>> pykov.Vector(A=.3, B=.7)
        {'A':.3, 'B':.7}
        """
        #dict.__init__(self)
        if data:
            self.update(data)
        if len(kwargs):
            self.update(kwargs)
    def __getitem__(self, key):
        """
        >>> q = pykov.Vector(C=.4, B=.6)
        >>> q['C']
        0.4
        >>> q['Z']
        0.0
        >>> 'Z' in q
        False
        """
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return 0.0
    def __setitem__(self, key, value):
        """
        >>> q = pykov.Vector(C=.4, B=.6)
        >>> q['Z']=.2
        >>> q
        {'C': 0.4, 'B': 0.6, 'Z': 0.2}
        >>> q['Z']=0
        >>> q
        {'C': 0.4, 'B': 0.6}
        """
        if value:
            dict.__setitem__(self, key, value)
        elif key in self:
            del(self[key])
    def __mul__(self, M):
        """
        >>> p = pykov.Vector(A=.3, B=.7)
        >>> p * 3
        {'A': 0.9, 'B': 2.1}
        >>> q = pykov.Vector(C=.5, B=.5)
        >>> p * q
        0.35
        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> p * T
        {'A': 0.91, 'B': 0.09}
        >>> T * p
        {'A': 0.42, 'B': 0.3}
        """
        if isinstance(M,int) or isinstance(M,float):
            result = Vector()
            for state, value in self.iteritems():
                result[state] = value * M
        if isinstance(M, Matrix):
            e2p, p2e = M._el2pos_() 
            x = self._toarray(e2p)
            y = numpy.zeros(len(x))
            M._ll_mat_(e2p).matvec_transp(x, y)
            result = Vector()
            result._fromarray(y, e2p)
            return result
        elif isinstance(M, Vector):
            result = 0
            for state, value in self.iteritems():
                result += value * M[state]
            return result
        else:
            raise TypeError('unsupported operand type(s) for *:'+
                            ' \'Vector\' and '+repr(type(M))[7:-1])
    def __rmul__(self, M):
        """
        >>> p = pykov.Vector(A=.3, B=.7)
        >>> 3 * p 
        {'A': 0.9, 'B': 2.1}
        """
        if isinstance(M,int) or isinstance(M,float):
            result = Vector()
            for state, value in self.iteritems():
                result[state] = value * M
            return result
        else:
            raise TypeError('unsupported operand type(s) for *: '+
                            repr(type(M))[7:-1]+' and \'Vector\'')
    def __add__(self, v):
        """
        >>> p = pykov.Vector(A=.3, B=.7)
        >>> q = pykov.Vector(C=.5, B=.5)
        >>> p + q
        {'A': 0.3, 'C': 0.5, 'B': 1.2}
        """
        if isinstance(v, Vector):
            result = Vector()
            for state in set(self.keys()) | set(v.keys()):
                result[state] = self[state] + v[state]
            return result
        else:
            raise TypeError('unsupported operand type(s) for +:'+
                            ' \'Vector\' and '+repr(type(v))[7:-1])
    def __sub__(self, v):
        """
        >>> p = pykov.Vector(A=.3, B=.7)
        >>> q = pykov.Vector(C=.5, B=.5)
        >>> p - q
        {'A': 0.3, 'C': -0.5, 'B': 0.2}
        >>> q - p
        {'A': -0.3, 'C': 0.5, 'B': -0.2}
        """
        if isinstance(v, Vector):
            result = Vector()
            for state in set(self.keys()) | set(v.keys()):
                result[state] = self[state] - v[state]
            return result
        else:
            raise TypeError('unsupported operand type(s) for -:'+
                            ' \'Vector\' and '+repr(type(v))[7:-1])
    def _toarray(self, el2pos):
        """
        >>> p = pykov.Vector(A=.3, B=.7)
        >>> el2pos = {'A': 1, 'B': 0}
        >>> v = p._toarray(el2pos)
        >>> v
        array([ 0.7,  0.3])
        """
        p = numpy.zeros(len(el2pos))
        for key, value in self.iteritems():
            p[el2pos[key]] = value
        return p
    def _fromarray(self, arr, el2pos):
        """
        >>> p = pykov.Vector()
        >>> el2pos = {'A': 1, 'B': 0}
        >>> v = numpy.array([ 0.7,  0.3])
        >>> p._fromarray(v,el2pos)
        >>> p
        {'A': 0.3, 'B': 0.7}
        """
        for elem, pos in el2pos.iteritems():
            self[elem] = arr[pos]
        return None
    def sort(self, reverse=False):
        """
        Sort according the probability.
        
        >>> p = pykov.Vector({'A':.3, 'B':.1, 'C':.6})
        >>> p.sort()
        [('B', 0.1), ('A', 0.3), ('C', 0.6)]
        >>> p.sort(reverse=True)
        [('C', 0.6), ('A', 0.3), ('B', 0.1)]
        """
        res = self.items()
        res.sort(key=lambda lst: lst[1], reverse=reverse)
        return res
    def normalize(self):
        """
        Normalize the vector so that the entries sum is 1.

        >>> p = pykov.Vector({'A':3, 'B':1, 'C':6})
        >>> p.normalize()
        >>> p
        {'A': 0.3, 'C': 0.6, 'B': 0.1}
        """
        s = self.sum()
        for k in self.keys():
            self[k] = self[k]/s
    def choose(self):
        """
        Choose a state according to its probability.

        >>> p = pykov.Vector(A=.3, B=.7)
        >>> p.choose()
        'B'

        .. seealso::

           `Kevin Parks recipe <http://code.activestate.com/recipes/117241/>`_
        """
        n = random.uniform(0, 1)
        for state, prob in self.iteritems():
            if n < prob:
                break
            n = n - prob
        return state
    def entropy(self):
        """
        Return the entropy.

        .. math::
        
           H(p) = \sum_i p_i \ln p_i

        .. seealso::
        
           Khinchin, A. I.
           Mathematical Foundations of Information Theory
           Dover, 1957.
        
        >>> p = pykov.Vector(A=.3, B=.7)
        >>> p.entropy()
        0.6108643020548935
        """
        return -sum([v*math.log(v) for v in self.itervalues()])
    def relative_entropy(self,p):
        """
        Return the Kullback-Leibler distance.
        
        .. math::
           
           d(q,p) = \sum_i q_i \ln (q_i/p_i)
        
        .. note::

           The Kullback-Leibler distance is not symmetric.

        >>> p = pykov.Vector(A=.3, B=.7)
        >>> q = pykov.Vector(A=.4, B=.6)
        >>> p.relative_entropy(q)
        0.02160085414354654
        >>> q.relative_entropy(p)
        0.022582421084357485
        """
        states = set(self.keys()) & set(p.keys())
        return sum([self[s]*math.log(self[s]/p[s]) for s in states])
    def copy(self):
        """
        Return a shallow copy.

        >>> p = pykov.Vector(A=.3, B=.7)
        >>> q = p.copy()
        >>> p['C'] = 1.
        >>> q
        {'A': 0.3, 'B': 0.7}
        """
        return Vector(self)
    def sum(self):
        """
        Sum the values.

        >>> p = pykov.Vector(A=.3, B=.7)
        >>> p.sum()
        1.0
        """
        return float(sum(self.itervalues()))
    def dist(self, p):
        """
        Return the distance between the two probability vectors.
        
        .. math::
        
           d(q,p) = \sum_i |q_i - p_i|
        
        >>> p = pykov.Vector(A=.3, B=.7)
        >>> q = pykov.Vector(C=.5, B=.5)
        >>> q.dist(p)
        1.0
        """
        if isinstance(v, Vector):
            result = 0
            for state in set(self.keys()) | set(v.keys()):
                result += abs(v[state]-self[state])
            return result

class Matrix(dict):
    """
    """

    def __init__(self, data=None):
        """
        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        """
        #dict.__init__(self)
        if data:
            self.update(data)
    def __getitem__(self, *args):
        """
        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T[('A','B')]
        0.3
        >>> T['A','B']
        0.3
        >>> T['B','B']
        0.0
        """
        try:
            return dict.__getitem__(self, args[0])
        except KeyError:
            return 0.0
    @_del_cache
    def __setitem__(self, key, value):
        """
        >>> T = pykov.Matrix()
        >>> T[('A','B')] = .3
        >>> T
        {('A', 'B'): 0.3}
        >>> T['A','A'] = .7
        >>> T
        {('A', 'B'): 0.3, ('A', 'A'): 0.7}
        >>> T['B','B'] = 0
        >>> T
        {('A', 'B'): 0.3, ('A', 'A'): 0.7}
        >>> T['A','A'] = 0
        >>> T
        {('A', 'B'): 0.3}

        >>> T = pykov.Matrix({('A','B'): 3, ('A','A'): 7, ('B','A'): .1})
        >>> T.states()
        set(['A', 'B'])
        >>> T['A','C']=1
        >>> T.states()
        set(['A', 'C', 'B'])
        >>> T['A','C']=0
        >>> T.states()
        set(['A', 'B'])
        """
        if value:
            dict.__setitem__(self, key, value)
        elif key in self:
            del(self[key])
    @_del_cache
    def __delitem__(self, key):
        """
        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> del(T['B', 'A'])
        >>> T
        {('A', 'B'): 0.3, ('A', 'A'): 0.7}
        """
        dict.__delitem__(self, key)
    @_del_cache
    def pop(self, key):
        """
        Remove specified key and return the corresponding value.
        See: help(dict.pop)

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.pop(('A','B'))
        0.3
        >>> T
        {('B', 'A'): 1.0, ('A', 'A'): 0.7}
        """
        return  dict.pop(self, key)
    @_del_cache
    def popitem(self):
        """
        Remove and return some (key, value) pair as a 2-tuple.
        See: help(dict.popitem)

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.popitem()
        (('B', 'A'), 1.0)
        >>> T
        {('A', 'B'): 0.3, ('A', 'A'): 0.7}
        """
        return  dict.popitem(self)
    @_del_cache
    def clear(self):
        """
        Remove all keys.
        See: help(dict.clear)

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.clear()
        >>> T
        {}
        """
        dict.clear(self)
    @_del_cache
    def update(self, other):
        """
        Update with keys and their values present in other.
        See: help(dict.update)

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> d = {('B', 'C'):2}
        >>> T.update(d)
        >>> T
        {('B', 'A'): 1.0, ('B', 'C'): 2, ('A', 'B'): 0.3, ('A', 'A'): 0.7}
        """
        dict.update(self, other)
    @_del_cache
    def setdefault(self, k, *args):
        """
        See: help(dict.setdefault)

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.setdefault(('A','A'),1)
        0.7
        >>> T
        {('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 0.7}
        >>> T.setdefault(('A','C'),1)
        1
        >>> T
        {('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 0.7, ('A', 'C'): 1}
        """
        return dict.setdefault(self, k, *args)
    def copy(self):
        """
        Return a shallow copy.

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> W = T.copy()
        >>> T[('B','B')] = 1.
        >>> W
        {('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 0.7}
        """
        return Matrix(self)
    def _ll_mat_(self, el2pos, method=''):
        """
        """
        m = len(el2pos)
        n = len(self)
        ll_mat = pysparse.spmatrix.ll_mat(m, m, n)
        if method == '':
            for k, v in self.iteritems():
                i = el2pos[k[0]]
                j = el2pos[k[1]]
                ll_mat[i,j] = float(v)
        elif method == 'transpose':
            for k, v in self.iteritems():
                i = el2pos[k[0]]
                j = el2pos[k[1]]
                ll_mat[j,i] = float(v)
        return ll_mat
    def _from_ll_mat_(self, mat, pos2el):
        """
        """
        for ii, val in mat.items():
            self[pos2el[ii[0]],pos2el[ii[1]]] = val
        return None
    def _numpy_mat(self, el2pos):
        """
        Return a numpy.matrix object from a dictionary.
    
        -- Parameters --
        t_ij : the dict, values must be real numbers, keys should be tuples of
        two strings.
        el2pos : see _map()
        """
        m = len(el2pos)
        T = numpy.matrix(numpy.zeros((m, m)))
        for k, v in self.iteritems():
            T[el2pos[k[0]], el2pos[k[1]]] = v
        return T
    
    def _from_numpy_mat(self, T, pos2el):
        """
        Return a dictionary from a numpy.matrix object.
    
        -- Parameters --
        T : the numpy.matrix.
        pos2el : see _map()
        """
        for i in range(len(T)):
            for j in range(len(T)):
                if T[i,j]:
                    self[(pos2el[i], pos2el[j])] = T[i, j]
        return None
    def _el2pos_(self):
        """
        """
        el2pos = {}
        pos2el = {}
        for pos,element in enumerate(list(self.states())):
            el2pos[element] = pos
            pos2el[pos] = element
        return el2pos, pos2el
    #def _pos2el_(self):
    #    """
    #    """
    #    pos2el = {}
    #    for pos,element in enumerate(list(self.states())):
    #        pos2el[pos] = element
    #    return pos2el
    def stochastic(self):
        """
        Make a right stochastic matrix.

        Set the sum of every row equal to one,
        raise ``PykovError`` if it is not possible.
        
        >>> T = pykov.Matrix({('A','B'): 3, ('A','A'): 7, ('B','A'): .2})
        >>> T.stochastic()
        >>> T
        {('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 0.7}
        >>> T[('A','C')]=1
        >>> T.stochastic()
        pykov.PykovError: 'Zero links from node C'
        """
        s = {}
        for k, v in self.succ().iteritems():
            summ = float(sum(v.values()))
            if summ:
                s[k] = summ
            else:
                raise PykovError('Zero links from state '+k)
        for k in self.keys():
            self[k] = self[k]/s[k[0]]
    def pred(self, key=None):
        """
        Return the precedessors of a state (if not indicated, of all states).
        In Matrix notation: return the coloum of the indicated state.

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.pred()
        {'A': {'A': 0.7, 'B': 1.0}, 'B': {'A': 0.3}}
        >>> T.pred('A')
        {'A': 0.7, 'B': 1.0}
        """
        try:
            if key:
                return self._pred[key]
            else:
                return self._pred
        except AttributeError:
            self._pred = dict([(state, Vector()) for state in self.states()])
            for link, probability in  self.iteritems():
                self._pred[link[1]][link[0]] = probability
            if key:
                return self._pred[key]
            else:
                return self._pred
    def succ(self, key=None):
        """
        Return the successors of a state (if not indicated, of all states).
        In Matrix notation: return the row of the indicated state.

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.succ()
        {'A': {'A': 0.7, 'B': 0.3}, 'B': {'A': 1.0}}
        >>> T.succ('A')
        {'A': 0.7, 'B': 0.3}
        """
        try:
            if key:
                return self._succ[key]
            else:
                return self._succ
        except AttributeError:
            self._succ = dict([(state, Vector()) for state in self.states()])
            for link, probability in  self.iteritems():
                self._succ[link[0]][link[1]] = probability
            if key:
                return self._succ[key]
            else:
                return self._succ
#    def remove(self, state):
#        """
#        Return a copy of the Chain, without the indicated state.
#        
#        All the links where the state appears are deleted, so that the result
#        will not be in general a stochastic matrix.
#
#        Example
#        -------
#        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
#        >>> Z = T.remove('B')
#        >>> Z
#        {('A', 'A'): 0.7}
#        """
#        return Matrix(dict([(key, value) for key, value in self.iteritems() if
#               state not in key]))
    def remove(self, states):
        """
        Return a copy of the Chain, without the indicated states.

        .. warning::
           
           All the links where the states appear are deleted, so that the result
           will not be in general a stochastic matrix.
        ..

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.remove(['B'])
        {('A', 'A'): 0.7}
        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.,
                             ('C','D'): .5, ('D','C'): 1., ('C','B'): .5})
        >>> T.remove_from(['A','B'])
        {('C', 'D'): 0.5, ('D', 'C'): 1.0}
        """
        return Matrix(dict([(key, value) for key, value in self.iteritems() if
                       key[0] not in states and key[1] not in states]))
#    @property
#    def states(self):
#        """
#        Example
#        -------
#        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
#        >>> T.states
#        set(['A', 'B'])
#        """
#        try:
#            return self.__dict__["states"]
#        except KeyError:
#            self.__dict__["states"] = set()
#            for link in  self.iterkeys():
#                self.states().add(link[0])
#                self.states().add(link[1])
#            return self.states()
    def states(self):
        """
        Return the set of states.

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.states()
        set(['A', 'B'])
        """
        try:
            return self._states
        except AttributeError:
            self._states = set()
            for link in  self.iterkeys():
                self._states.add(link[0])
                self._states.add(link[1])
            return self._states
    def __mul__(self, v):
        """
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
        """
        if isinstance(v, Vector):
            e2p, p2e = self._el2pos_()
            x = v._toarray(e2p)
            y = numpy.zeros(len(x))
            self._ll_mat_(e2p).matvec(x, y)
            result = Vector()
            result._fromarray(y, e2p)
            return result
        elif isinstance(v, Matrix):
            e2p, p2e = self._el2pos_()
            C = pysparse.spmatrix.matrixmultiply(self._ll_mat_(e2p), v._ll_mat_(e2p))
            if 'Chain' in repr(self.__class__):
                res = Chain()
            elif 'Matrix' in repr(self.__class__):
                res = Matrix()
            res._from_ll_mat_(C, p2e)
            return res
        elif isinstance(v,int) or isinstance(v,float):
            return  Matrix(dict([(key, value * v) for key, value in
                    self.iteritems()]))
        else:
            raise TypeError('unsupported operand type(s) for *:'+
                            ' \'Matrix\' and '+repr(type(v))[7:-1])
    def __rmul__(self, v):
        """
        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> 3 * T
        {('B', 'A'): 3.0, ('A', 'B'): 0.9, ('A', 'A'): 2.1}
        """
        if isinstance(v,int) or isinstance(v,float):
            return  Matrix(dict([(key, value * v) for key, value in
                    self.iteritems()]))
        else:
            raise TypeError('unsupported operand type(s) for *:'+
                            ' \'Matrix\' and '+repr(type(v))[7:-1])
    def __add__(self, M):
        """
        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> I = pykov.Matrix({('A','A'):1, ('B','B'):1})
        >>> T + I
        {('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 1.7, ('B', 'B'): 1.0}
        """
        if isinstance(M, Matrix):
            result = Matrix()
            for link in set(self.keys()) | set(M.keys()):
                result[link] = self[link] + M[link]
            return result
        else:
            raise TypeError('unsupported operand type(s) for +:'+
                            ' \'Matrix\' and '+repr(type(M))[7:-1])
    def __sub__(self, M):
        """
        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> I = pykov.Matrix({('A','A'):1, ('B','B'):1})
        >>> T - I
        {('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): -0.3, ('B', 'B'): -1}
        """
        if isinstance(M, Matrix):
            result = Matrix()
            for link in set(self.keys()) | set(M.keys()):
                result[link] = self[link] - M[link]
            return result
        else:
            raise TypeError('unsupported operand type(s) for -:'+
                            ' \'Matrix\' and '+repr(type(M))[7:-1])
    def trace(self):
        """
        Return the Matrix trace.

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.trace()
        0.7
        """
        return sum([self[k,k] for k in self.states()])
    def eye(self):
        """
        Return the Identity Matrix.

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.eye()
        {('A', 'A'): 1., ('B', 'B'): 1.}
        """
        return Matrix(dict([((state, state), 1.) for state in self.states()]))
    def ones(self):
        """
        Return a ``Vector`` instance with entries equal to one.

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.ones()
        {'A': 1.0, 'B': 1.0}
        """
        return Vector(dict([(state, 1.) for state in self.states()]))
    def transpose(self):
        """
        Return the transpose Matrix.

        >>> T = pykov.Matrix({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.transpose()
        {('B', 'A'): 0.3, ('A', 'B'): 1.0, ('A', 'A'): 0.7}
        """
        return Matrix(dict([((key[1], key[0]), value) for key, value in
               self.iteritems()]))
    def _BiCGSTAB(self, b, x=None, error=1e-15, maxit=int(1e9)):
        """
        Biconjugate gradient stabilized method.
        A * x = b

        References
        ----------
        Van der Vorst, H. A. (1992). "Bi-CGSTAB: A Fast and Smoothly Converging
        Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems". SIAM
        Journal on Scientific and Statistical Computing 13: 631–644.
        """
        if not x:
            x = numpy.ones(len(self.states()))
        e2p, p2e = self._el2pos_()
        A = self._ll_mat_(e2p).to_csr()
        bb = b._toarray(e2p)
        info, niter, relres = pysparse.itsolvers.bicgstab(A, bb, x, error, maxit)
        if info > 0:
            print('Sorry, not converged.')
            return None
        res = Vector()
        res._fromarray(x, e2p)
        return res
    def _UMPFPACKSolve(self, b, x=None, method='UMFPACK_A'):
        """
        UMFPACK ( U nsymmetric M ulti F Rontal PACK age)
        
        Parameters
        ----------
        method (see pysparse doc.):
            "UMFPACK_A"     :  \mathbf{A} x = b (default)
            "UMFPACK_At"    :  \mathbf{A}^T x = b
            "UMFPACK_Pt_L"  :  \mathbf{P}^T \mathbf{L} x = b
            "UMFPACK_L"     :  \mathbf{L} x = b
            "UMFPACK_Lt_P"  :  \mathbf{L}^T \mathbf{P} x = b
            "UMFPACK_Lt"    :  \mathbf{L}^T x = b
            "UMFPACK_U_Qt"  :  \mathbf{U} \mathbf{Q}^T x = b
            "UMFPACK_U"     :  \mathbf{U} x = b
            "UMFPACK_Q_Ut   :  \mathbf{Q} \mathbf{U}^T x = b
            "UMFPACK_Ut"    :  \mathbf{U}^T x = b

        References
        ----------
        A column pre-ordering strategy for the unsymmetric-pattern multifrontal
        method, T. A. Davis, ACM Transactions on Mathematical Software, vol 30,
        no. 2, June 2004, pp. 165-195.
        """
        if not x:
            x = numpy.ones(len(self.states()))
        e2p, p2e = self._el2pos_()
        A = self._ll_mat_(e2p)
        bb = b._toarray(e2p)
        LU = pysparse.umfpack.factorize(A)
        LU.solve(bb, x, method)
        res = Vector()
        res._fromarray(x, e2p)
        return res

class Chain(Matrix):
    """
    """   
    
    def move(self, state):
        """
        Do one step from the indicated state, and return the final state.

        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.move('A')
        'B'
        """
        return self.succ(state).choose() 
    def pow(self, p, n):
        """
        Find the probability distribution after n steps, starting from an
        initial ``Vector``.

        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> p = pykov.Vector(A=1)
        >>> T.pow(p,3)
        {'A': 0.7629999999999999, 'B': 0.23699999999999996}
        >>> p * T * T * T
        {'A': 0.7629999999999999, 'B': 0.23699999999999996}
        """
        e2p, p2e = self._el2pos_()
        A = self._ll_mat_(e2p).to_csr()
        x = p._toarray(e2p)
        y = numpy.zeros(len(e2p))
        for i in xrange(n):
            A.matvec_transp(x, y)
            x = y.copy()
        res = Vector()
        res._fromarray(y, e2p)
        return res
    #def vector(self, data=None, **kwargs):
    #    """
    #    Example
    #    ------
    #    See __init__ method of Vector class. 
    #    >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
    #    >>> p = T.vector(A=1)
    #    """
    #    if not data:
    #        data = {}
    #    data.update(kwargs)
    #    res = Vector(data)
    #    res._chain = self
    #    return res
#    def _GaussSeidel(self, p=None, error=1e-12):
#        """
#        """
#        m = len(self.states())
#        Q = self.eye() - self
#        Q = Q.transpose()
#        e2p, p2e = self._el2pos_()
#        A = Q._ll_mat_(e2p)
#        D_E = pysparse.spmatrix.ll_mat(m, m, m) #diagonal + lower diagonal part
#        F = pysparse.spmatrix.ll_mat(m, m, m)   #upper diagonal part
#        for k, v in A.items():
#            if k[0] >= k[1]:
#                D_E[k] = v
#            else:
#                F[k] = -v
#        F = F.to_csr()
#        LU = pysparse.umfpack.factorize(D_E)
#        if not p:
#            if hasattr(self,'_guess'):
#                p = self._guess
#            else:
#                p = Vector({random.sample(self.states(),1)[0]:1.})
#        x = p._toarray(e2p)
#        err = 1.
#        n = 0
#        xold = x.copy()
#        b = numpy.zeros(m)
#        while err > error :
#            F.matvec(x, b)
#            LU.solve(b, x)
#            n += 1
#            if not n % 1000:
#                err = numpy.linalg.norm( xold - x )
#                print err
#                xold = x.copy()
#        res = Vector()
#        res._fromarray(x, e2p)
#        res.normalize()
#        self._steady = res
#        return res
#    def steady_old(self):
#        """
#        With the assumption of ergodicity, return the steady state.
#        
#        Maths
#        -----
#        Inverse iteration method:
#        Q = T - 11
#        Q.trasp * x = e_n
#
#        References
#        ----------
#        W. Stewart: Introduction to the Numerical Solution of Markov Chains,
#        Princeton University Press, Chichester, West Sussex, 1994.
#
#        Example
#        -------
#        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
#        >>> T.steady()
#        {'A': 0.7692307692307676, 'B': 0.23076923076923028}
#        """
#        try:
#            return self._steady
#        except AttributeError:
#            e2p, p2e = self._el2pos_()
#            M = self._ll_mat_(e2p)
#            m = len(self.states())
#            r = range(m)
#            M.put([M[i,i]-1. for i in r],r,r)
#            b = numpy.zeros(m)
#            x = numpy.zeros(m)
#            b[-1] = 1.
#            LU = pysparse.umfpack.factorize(M)
#            LU.solve(b, x, "UMFPACK_At")
#            res = Vector()
#            res._fromarray(x, e2p)
#            res.normalize()
#            self._steady = res
#            return res
    def steady(self):
        """
        With the assumption of ergodicity, return the steady state.
        
        .. note::
           
           Inverse iteration method (P is the Markov chain)

           .. math::
           
              Q = \mathbf{I} - P

              Q^T x = e

              e = (0,0,\dots,0,1)
           ..
        ..


        .. seealso::

           W. Stewart: Introduction to the Numerical Solution of Markov Chains,
           Princeton University Press, Chichester, West Sussex, 1994.

        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.steady()
        {'A': 0.7692307692307676, 'B': 0.23076923076923028}
        """
        try:
            return self._steady
        except AttributeError:
            e2p, p2e = self._el2pos_()
            M = self._ll_mat_(e2p, 'transpose')
            M.scale(-1)
            m = len(self.states())
            r = range(m)
            M.put([M[i,i]+1. for i in r],r,r)
            b = numpy.zeros(m)
            x = numpy.zeros(m)
            b[-1] = 1.
            try:
                LU = pysparse.umfpack.factorize(M)
            except SystemError:  #not elegant this singular matrix error..
                # really dirty trick ??
                M[0,0] = M[0,0] + _machineEpsilon()
                LU = pysparse.umfpack.factorize(M)
            LU.solve(b, x)
            res = Vector()
            res._fromarray(x, e2p)
            res.normalize()
            self._steady = res
            return res
    def entropy(self, p=None, norm=False):
        """
        Return the ``Chain`` entropy, calculated with the indicated probability
        Vector (the steady state by default).

        .. math::
        
           H_i = \sum_j P_{ij} \ln P_{ij}
           
           H = \sum \pi_i  H_i

        .. seealso::
        
           Khinchin, A. I.
           Mathematical Foundations of Information Theory
           Dover, 1957.

        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.entropy()
        0.46989561696530169

        With normalization entropy belongs to [0,1]
        
        >>> T.entropy(norm=True)
        0.33895603665233132

        """
        if not p:
            p = self.steady()
        H = 0.
        for state in self.states():
            H += p[state] * sum([v * math.log(v) for v in
                                self.succ(state).itervalues()])
        if norm:
            n = len(self.states())
            return -H / (n * math.log(n))
        return -H
    def mfpt_to(self, state):
    #def mfpt_to(self, state, error=1e-08, guess=None):
        """
        Return the Mean First Passage Times of every state to the indicated
        state.

        .. seealso::

           Kemeny J. G.; Snell, J. L.
           Finite Markov Chains.
           Springer-Verlag: New York, 1976.

        >>> d = {('R', 'N'): 0.25, ('R', 'S'): 0.25, ('S', 'R'): 0.25,
                 ('R', 'R'): 0.5, ('N', 'S'): 0.5, ('S', 'S'): 0.5,
                 ('S', 'N'): 0.25, ('N', 'R'): 0.5, ('N', 'N'): 0.0}
        >>> T = pykov.Chain(d)
        >>> T.mfpt_to('R')
        {'S': 3.333333333333333, 'N': 2.666666666666667}
        """
        if len(self.states()) == 2:
            self.states().remove(state)
            other = self.states().pop()
            self.states().add(state)
            self.states().add(other)
            return Vector({other: 1. / self[other,state]})
        T = self.remove([state])
        T = T.eye() - T
        return T._UMPFPACKSolve(T.ones())
        #return  T._BiCGSTAB(T.ones(), error=error, x=guess)
    def adiacence(self):
        """
        Return the adiacence matrix.

        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.adiacence()
        {('B', 'A'): 1, ('A', 'B'): 1, ('A', 'A'): 1}
        """
        return Matrix(dict.fromkeys(self,1))
    def walk(self, steps, start=None, stop=None):
        """
        Return a random walk of n steps, starting and stopping at the
        indicated states.
        
        .. note::

           If not indicated, then the starting state is chosen according
           to its steady probability.
           If the stopping state is reached before to do n steps, then the walker
           stops.

        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.walk(10)
        ['B', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A']
        >>> T.walk(10,'B','B')
        ['B', 'A', 'A', 'A', 'A', 'A', 'B']
        """
        if not start:
            start = self.steady().choose()
        if not stop:
            result = [start]
            for i in range(steps):
                result.append(self.move(result[-1]))
            return result
        if stop:
            result = [start]
            for i in range(steps):
                result.append(self.move(result[-1]))
                if result[-1] == stop:
                    return result
            return result
    def walk_probability(self, walk):
        """
        Given a walk, return the log of its probability.

        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.walk_probability(['A','A','B','A','A'])
        -1.917322692203401
        >>> probability = math.exp(-1.917322692203401)
        0.147
        >>> p = T.walk_probability(['A','B','B','B','A'])
        >>> math.exp(p)
        0.0
        """
        res = 0
        for step in zip(walk[:-1], walk[1:]):
            if not self[step]:
                return -float('Inf')
            res += math.log(self[step])
        return res
#    def not_reversible(self):
#        """
#        (Heuristic) Return a float value between 1 and 0, where 0 means that the Chain is reversible.
#        """
#        res = 0
#        S = set()
#        pi = self.steady()
#        for link in self:
#            if link not in S:
#                S.add(link)
#                S.add((link[1],link[0]))
#                res += abs(pi[link[0]] * self[link] -
#                           pi[link[1]] * self[(link[1],link[0])]
#                          )
#        return res
    def mixing_time(self, cutoff=.25, jump=1, p=None):
        """
        Return the mixing time.

        If the initial distribution (p) is not indicated,
        then it is set to p={'less probable state':1}.

        .. note::
        
           The mixing time is calculated here as the number of steps (n) needed to
           have

           .. math::

              |p(n)-\pi| < 0.25
              
              p(n)=p P^n
              
              \pi=\pi P
           ..

           The parameter ``jump`` controls the iteration step, for example with
           ``jump=2`` n has values 2,4,6,8,..
        ..

        >>> d = {('R','R'):1./2, ('R','N'):1./4, ('R','S'):1./4,
                 ('N','R'):1./2, ('N','N'):0., ('N','S'):1./2,
                 ('S','R'):1./4, ('S','N'):1./4, ('S','S'):1./2}
        >>> T = pykov.Chain(d)
        >>> T.mixing_time()
        2
        """
        res = []
        d = 1
        n = 0
        if not p:
            p = Vector({self.steady().sort()[0][0]:1})
        res.append(p.dist(self.steady()))
        ##res.append(p.relative_entropy(self.steady()))
        while d > cutoff:
            n = n + jump
            p = self.pow(p,jump)
            d = p.dist(self.steady())
            ##d = p.relative_entropy(self.steady())
            res.append(d)
        #return n, [range(0,n+1,jump),res]
        return n
    def absorbing_time(self, transient_set):
        """
        Mean number of steps needed to leave the transient set.

        Return the ``Vector tau``, the ``tau[i]`` is the mean number of steps needed
        to leave the transient set starting from state ``i``. The parameter
        ``transient_set`` is a subset of nodes.
        
        .. note::
           
           If the starting point is a ``Vector p``, then it is sufficient to
           calculate ``p * tau`` in order to weigh the mean times according the
           initial conditions.


        .. seealso:
           
           Kemeny J. G.; Snell, J. L. 
           Finite Markov Chains.
           Springer-Verlag: New York, 1976.

        >>> d = {('R','R'):1./2, ('R','N'):1./4, ('R','S'):1./4,
                 ('N','R'):1./2, ('N','N'):0., ('N','S'):1./2,
                 ('S','R'):1./4, ('S','N'):1./4, ('S','S'):1./2}
        >>> T = pykov.Chain(d)
        >>> p = pykov.Vector({'N':.3, 'S':.7})
        >>> tau = T.absorbing_time(p.keys())
        >>> p * tau
        3.1333333333333329
        """
        Q = self.remove(self.states() - set(transient_set)) 
        K = Q.eye() - Q
        #means
        tau = K._UMPFPACKSolve(K.ones())
        return tau
        #-----
        #variances
        #Ntau = K._UMPFPACKSolve(tau)
        #tau_square = Vector({})
        #for k,v in tau.iteritems():
        #    tau_square[k] = v**2
        #var_tau = 2 * Ntau - tau - tau_square
        #--------
        #return tau, var_tau
    def absorbing_tour(self, p, transient_set=None):
        """
        Return a ``Vector v``, ``v[i]`` is the mean of the total number of times
        the process is in a given transient state ``i`` before to leave the
        transient set.
        
        .. note::
           ``v.sum()`` is equal to ``p * tau`` (see :meth:`absorbing_time` method).

        In not specified, the ``transient set`` (with its probability) is defined
        by means of the ``Vector p``.

        .. seealso::

           Kemeny J. G.; Snell, J. L. 
           Finite Markov Chains.
           Springer-Verlag: New York, 1976.
        
        >>> d = {('R','R'):1./2, ('R','N'):1./4, ('R','S'):1./4,
                 ('N','R'):1./2, ('N','N'):0., ('N','S'):1./2,
                 ('S','R'):1./4, ('S','N'):1./4, ('S','S'):1./2}
        >>> T = pykov.Chain(d)
        >>> p = pykov.Vector({'N':.3, 'S':.7})
        >>> T.absorbing_tour(p)
        {'S': 2.2666666666666666, 'N': 0.8666666666666669}
        """
        if transient_set:
            Q = self.remove(self.states() - transient_set) 
        else:
            Q = self.remove(self.states() - set(p.keys()))
        K = Q.eye() - Q
        return K._UMPFPACKSolve(p,method='UMFPACK_At')
    def fundamental_matrix(self):
        """
        Return the fundamental matrix.

        .. seealso::
        
           Kemeny J. G.; Snell, J. L. 
           Finite Markov Chains.
           Springer-Verlag: New York, 1976.
        
        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.fundamental_matrix()
        {('B', 'A'): 0.17751479289940991, ('A', 'B'): 0.053254437869822958,
        ('A', 'A'): 0.94674556213017902, ('B', 'B'): 0.82248520710059214}
        """
        try:
            return self._fundamental_matrix
        except AttributeError:
            el2pos, pos2el = self._el2pos_()
            p = self.steady()._toarray(el2pos)        
            P = self._numpy_mat(el2pos)
            d = len(p)
            A = numpy.matrix([p for i in range(d)])
            I = numpy.matrix(numpy.identity(d))
            E = numpy.matrix(numpy.ones((d, d)))
            D = numpy.zeros((d, d))
            diag = 1. / p
            for pos,val in enumerate(diag):
                D[pos,pos] = val
            Z = numpy.linalg.inv(I - P + A)
            res = Matrix()
            res._from_numpy_mat(Z,pos2el)
            self._fundamental_matrix = res
            return res
    def kemeny_constant(self):
        """
        Return the Kemeny constant of the transition matrix.
        
        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
        >>> T.Kemeny_constant()
        1.7692307692307712
        """
        Z = self.fundamental_matrix()
        return Z.trace()
        #K = 0.
        #for state in self.states():
        #    K = K + Z.get((state, state), 0)
        #return K
#    def force_detailed_balance(self):
#        """
#        (Heuristic) Return a new Chain, which satisfies detailed balance.
#
#        Maths
#        -----
#        q_ij_db = q_ji_db = (pi_i * T_ij + pi_j * T_ji) / 2
#        pi_i_db = sum_j q_ij_db
#        Tij_db = q_ij_db / pi_i_db
#        """
#        tij = {}
#        pi = self.steady()
#        for key in self.iterkeys():
#            if key not in tij:
#                num = (pi[key[0]] * self[key] +
#                       pi[key[1]] * self[(key[1],key[0])])
#                den = (pi[key[0]] +
#                       sum([pi[k] * v for k,v in self.pred(key[0]).iteritems()]))
#                tij[key] = num/den
#                den = (pi[key[1]] +
#                       sum([pi[k] * v for k,v in self.pred(key[1]).iteritems()]))
#                tij[(key[1],key[0])] = num/den
#        return Chain(tij)
#    def cFEP(self, node, p=None, temp=None):
#        """
#        (Heuristic) Return the cut based free energy profile.
#    
#        Mfpt ordering.
#        Return a list of lists:
#        [[node, mfpt, p_A, p_AB, -KTlog(P_AB), -KTlog(P_AB/P_A)],
#        [...],
#        ...]
#    
#        -- Parameters --
#        chain : a Chain object
#        p_i : the steady state of the chain
#        node : the reference node
#        temp : the temperature (if None, KT is set to 1)
#        """
#        if not temp:
#            KT = 1
#        else:
#            KB = 1.987E-3 #Boltzman constant in kCal/(mol K)
#            KT=temp*KB
#        if not p:
#            p = self.steady()
#        #p_ij = {}
#        #for k, v in self.iteritems():
#        #    p_ij[k] = p[k[0]] * v
#        mfptnode = self.mfpt_to(node)
#        result = [[node, 0.]]
#        result.extend([list(i) for i in mfptnode.sort()])
#        A = set()
#        C = set()
#        p_A = 0.
#        for data in result[:-1]:
#            p_A = p_A +p[data[0]]
#            A = A | set([data[0]])
#            C = C | set(self.succ(data[0]).keys())
#            C = C - A
#            p_AB = 0.
#            for i in C:
#                for j in set(self.pred(i).keys()):
#                    if j in A:
#                        #p_AB = p_AB + p_ij[(j,i)]
#                        p_AB = p_AB + p[j] * self[j,i]
#            data.append(p_A)
#            data.append(p_AB)
#            data.append(-KT * math.log(p_AB))
#            data.append(-KT * math.log(p_AB / p_A))
#        return result
#    def graph(self, attribute_data=None):
#        """
#        Return the associated directed graph as a networkx.DiGraph object.
#        
#        Paramenters
#        -----------
#        attribute : a dict keyed by links an valued by an python objects.
#
#        Example
#        -------
#        >>> T = pykov.Chain({('A','B'): .3, ('A','A'): .7, ('B','A'): 1.})
#        >>> G = T.graph()
#        >>> G.edges()
#        [('A', 'A'), ('A', 'B'), ('B', 'A')]
#        """
#        G = networkx.DiGraph()
#        if not attribute_data:
#            for k1, k2 in self:
#                G.add_edge(k1, k2)
#        else:
#            for k1, k2 in self:
#                G.add_edge(k1, k2, attribute = attribute_data[(k1,k2)])
#        return G
#    def irreducible_components(self):
#        """
#        Return the irredicible (i.e. strongly connected) components, ordered in size.
#
#        Example
#        -------
#        >>> d = {('R','R'):1./2, ('R','N'):1./4, ('R','S'):1./4,
#                  ('N','N'):1./2, ('N','S'):1./2, ('S','N'):1./2, ('S','S'):1./2}
#        >>> T = pykov.Chain(d)
#        >>> T.irredicible_components()
#        [['N', 'S'], ['R']]
#
#        References
#        ----------
#        The function calls networkx.strongly_connected_components().
#        """
#        G = self.graph()
#        return networkx.strongly_connected_components(G)
#    def extract_irreducible_component(self,component):
#        """
#        Return the Chain associated to the indicated irreducible component.
#
#        An operation of normalization is imposed (see stochastic method).
#        """
#        res = self.remove(self.states() - set(component))
#        res = Chain(res)
#        res.stochastic()
#        return res
#    def test_detailed_balance(self):
#        """
#        """
#        p = self.steady()
#        result = 0
#        for k1, k2 in self.iterkeys():
#            result += abs(p[k1] * T[k1,k2] - p[k2] * T[k2,k1])
#        return result/2.
#    def fundamental_matrix_col(self, state):
#        """
#        """
#        A = Matrix()
#        for i in self.states():
#            for j in self.states():
#                A[i,j] = self.steady()[j]
#        L = self.eye() - self + A
#        x = L._BiCGSTAB(Vector({state:1.}), error=1e-08, x=None)
#        return x
#    def fundamental_matrix_row(self, state):
#        """
#        """
#        A = Matrix()
#        for i in self.states():
#            for j in self.states():
#                A[i,j] = self.steady()[j]
#        L = self.eye() - self + A
#        L = L.transpose()
#        x = L._BiCGSTAB(Vector({state:1.}), error=1e-08, x=None)
#        return x
#    def mfpt_to1(self, state):
#        """
#        """
#        pi = self.steady()
#        Z_state = self.fundamental_matrix_col(state)
#        result = Vector({i:Z_state[state]-Z_state[i] for i in Z_state})
#        return Vector({state:1./pi[state]}) + 1./pi[state] * result
#

def readmat(filename):
    """
    Read an external file and return a Chain.

    The file must be of the form:

    A A .7
    A B .3
    B A 1

    Example
    -------
    >>> P = pykov.readmat('/mypath/mat')
    >>> P
    {('B', 'A'): 1.0, ('A', 'B'): 0.3, ('A', 'A'): 0.7}
    """
    with open(filename) as f:
        P = Chain()
        for line in f:
            tmp = line.split()
            P[(tmp[0],tmp[1])] = float(tmp[2])    
        return P
def readtrj(filename):
    """
    In the case the :class:`Chain` instance must be created from a finite chain of
    states, the transition matrix is not fully defined.
    The function defines the transition probabilities as the maximum likelihood
    probabilities calculated along the chain. Having the file ``/mypath/trj`` with the following
    format::
        
        1
        1
        1
        2
        1
        3
        
    the :class:`Chain` instance defined from that chain is:

    >>> t = pykov.readtrj('/mypath/trj')
    >>> t
    (1, 1, 1, 2, 1, 3)
    >>> p, P = maximum_likelihood_probabilities(t,lag_time=1, separator='0')
    >>> p
    {1: 0.6666666666666666, 2: 0.16666666666666666, 3: 0.16666666666666666}
    >>> P
    {(1, 2): 0.25, (1, 3): 0.25, (1, 1): 0.5, (2, 1): 1.0, (3, 3): 1.0}
    >>> type(P)
    <class 'pykov.Chain'>
    >>> type(p)
    <class 'pykov.Vector'>
    """
    with open(filename) as f:
        return tuple(line.strip() for line in f)
def _writefile(mylist, filename):
    """
    Export in a file the list.

    mylist could be a list of list.

    Example
    -------
    >>> L = [[2,3],[4,5]]
    >>> pykov.writefile(L,'tmp')
    >>> l = [1,2]
    >>> pykov.writefile(l,'tmp')
    """
    try:
        L = [[str(i) for i in line] for line in mylist]
    except TypeError:
        L = [str(i) for i in mylist]
    with open(filename, mode='w') as f:
        tmp = '\n'.join('\t'.join(x) for x in L)
        f.write(tmp)
    return None
def transitions(trj, nsteps=1, lag_time=1, separator='0'):
    """
    Return the temporal list of transitions observed.

    Parameters
    ----------
    trj : the symbolic trajectory.
    nsteps : number of steps.
    lag_time : step length.
    separator: the special symbol indicating the presence of sub-trajectories.
    
    Example
    -------
    >>> trj = [1,2,1,0,2,3,1,0,2,3,2,3,1,2,3]
    >>> pykov.transitions(trj,1,1,0)
    [(1, 2), (2, 1), (2, 3), (3, 1), (2, 3), (3, 2), (2, 3), (3, 1), (1, 2),
    (2, 3)]
    >>> pykov.transitions(trj,1,2,0)
    [(1, 1), (2, 1), (2, 2), (3, 3), (2, 1), (3, 2), (1, 3)]
    >>> pykov.transitions(trj,2,2,0)
    [(2, 2, 1), (3, 3, 2), (2, 1, 3)]
    """
    result = []
    for pos in range( len(trj) - nsteps * lag_time ):
        if separator not in trj[pos:(pos + nsteps * lag_time + 1)]:
            tmp = trj[pos:(pos + nsteps * lag_time +1):lag_time]
            result.append(tuple(tmp))
    return result
def maximum_likelihood_probabilities(trj, lag_time=1, separator='0'):
    """
    Return a Chain calculated by means of maximum likelihood probabilities.
    
    Return two objects:
    p : a Vector object, the probability distribution over the nodes.
    T : a Chain object, the Markov chain.

    Parameters
    ----------
    trj : the symbolic trajectory.
    lag_time : number of steps defining a transition.
    separator: the special symbol indicating the presence of sub-trajectories.

    Example
    -------
    >>> t = [1,2,3,2,3,2,1,2,2,3,3,2]
    >>> p, T = pykov.maximum_likelihood_probabilities(t)
    >>> p
    {1: 0.18181818181818182, 2: 0.4545454545454546, 3: 0.36363636363636365}
    >>> T
    {(1, 2): 1.0, (3, 2): 0.7499999999999999, (2, 3): 0.5999999999999999, (3,
    3): 0.25, (2, 2): 0.19999999999999998, (2, 1): 0.19999999999999998}
    """
    q_ij = {}
    tr = transitions(trj, nsteps=1, lag_time=lag_time, separator=separator)
    _remove_dead_branch(tr)
    tot = len(tr)
    for step in tr:
            q_ij[step] = q_ij.get(step, 0.) + 1
    for key in q_ij.iterkeys():
        q_ij[key] = q_ij[key] / tot
    p_i = {}
    for k, v in q_ij.iteritems():
        p_i[k[0]] = p_i.get(k[0], 0) + v
    t_ij = {}
    for k, v in q_ij.iteritems():
        t_ij[k] = v / p_i[k[0]]
    T = Chain(t_ij)
    p = Vector(p_i)
    T._guess = Vector(p_i)
    return p, T
#def non_Markovian_flux(trj, lag_time=1, separator='0'):
#    """
#    (Heuristic) Return the non-Markovian flux.
#
#    It is a number between 0 and 1.
#    0 implies markovianity
#    1 implies not markovianity
#
#    Example
#    -------
#    >>> t = [1,2,3,2,3,2,1,2,2,3,3,2]
#    >>> pykov.non_Markovian_flux(t)
#    0.15000000000000013
#    """
#    p, C = maximum_likelihood_probabilities(trj,
#                   lag_time=lag_time, separator=separator)
#    tr = set(transitions(trj, nsteps=2, lag_time=lag_time,
#                  separator=separator))
#    r = [p[i[0]] * C[i[0], i[1]]  * C[i[1], i[2]] for i in tr]
#    return 1 - sum(r)
def _remove_dead_branch(transitions_list):
    """
    Remove dead branchs inserting a selfloop in every node that has not
    outgoing links.
    
    Example
    -------
    >>> trj = [1,2,3,1,2,3,2,2,4,3,5]
    >>> tr = pykov.transitions(trj, nsteps=1)
    >>> tr
    [(1, 2), (2, 3), (3, 1), (1, 2), (2, 3), (3, 2), (2, 2), (2, 4), (4, 3),
    (3, 5)]
    >>> pykov._remove_dead_branch(tr)
    >>> tr
    [(1, 2), (2, 3), (3, 1), (1, 2), (2, 3), (3, 2), (2, 2), (2, 4), (4, 3),
    (3, 5), (5, 5)]
    """
    head_set = set()
    tail_set = set()
    for step in transitions_list:
        head_set.add(step[1])
        tail_set.add(step[0])
    for head in head_set:
        if head not in tail_set:
            transitions_list.append((head, head))
    return None
def _machineEpsilon(func=float):
    machine_epsilon = func(1)
    while func(1)+func(machine_epsilon) != func(1):
        machine_epsilon_last = machine_epsilon
        machine_epsilon = func(machine_epsilon) / func(2)
    return machine_epsilon_last
#def minfloat(guess=1.):
#    while(guess * 0.5 != 0):
#        guess = guess * 0.5
#    return guess
