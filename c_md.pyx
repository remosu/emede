#! /usr/bin/env python

# cython: profile=True

import sys
import numpy as np
cimport numpy as np
from itertools import combinations, product


cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def cubic_lattice(L, D, N):
    n = np.ceil(N**(1.0 / D))
    l = L / n
    ax = np.linspace(0, L, n, endpoint=False)
    x0 = np.array([l/2.0] * D)
    result = np.array(list(product(ax, repeat=D))) + x0
    rr = range(len(result))
    indices = []
    for i in range(len(result)-N):
        pos = np.random.randint(len(rr)-i)
        indices.append(rr[pos])
        rr[pos], rr[len(rr)-i-1] = rr[len(rr)-i-1], rr[pos]  
    return np.delete(result, indices, axis=0)
    
cdef inline np.float_t LJ(np.float_t r2):
    cdef np.float_t ir2 = 1.0 / r2
    cdef np.float_t ir6 = ir2 * ir2 * ir2
    return 48 * ir2 * ir6 * (ir6 - 0.5)
    
cdef inline mic(np.ndarray r, double L, double cutoff):
    cdef int d = r.shape[0]
    cdef Py_ssize_t i
    for i in xrange(d):
        if r[i] > L -cutoff: #/ 2.0:
            r[i] -= L
        elif r[i] < cutoff - L: #- L / 2.0:
            r[i] += L
    
cdef class MD(object):
    cdef double size
    cdef int D
    cdef int nparts
    cdef double cutoff, cutoff2, ecut, dt, _e, _ek, _ep, _temp
    #~ cdef np.ndarray[DTYPE_t, ndim=2] positions, velocities, forces
    cdef public np.ndarray positions, velocities, forces
    def __init__(self, size=10, D=2, nparts=32, T=1, cutoff=4.0, dt=0.001):
        D = int(D)
        nparts = int(nparts)
        self.size = size
        self.D = D
        self.nparts = nparts
        self.cutoff = cutoff
        self.cutoff2 = self.cutoff**2
        self.ecut = 4.0 * (1.0 / cutoff**12 - 1.0 / cutoff**6)
        self.dt = dt
        self.positions = cubic_lattice(size, D, nparts)
        self.velocities = np.random.random((nparts,D))
        vcm = np.sum(self.velocities, axis=0) / nparts
        self.velocities -= vcm
        v2 = np.sum(self.velocities**2) / nparts
        self.velocities *= np.sqrt(D * T / v2)
        self.forces = np.zeros((nparts,D))
        self._start_acums()
    
    cdef _start_acums(self):
        self._e = 0.0
        self._ek = 0.0
        self._ep = 0.0
        self._temp = 0.0
        
    @property
    def N(self):
        return self.nparts
        
    @property
    def T(self):
        if self._temp == 0.0:
            self._temp = np.sum(self.velocities**2) / (self.D * len(self.velocities))
        return self._temp
        
    @property
    def EK(self):
        if self._ek == 0.0:
            self._ek = 0.5 * np.sum(self.velocities**2)
        return self._ek
        
    @property
    def EP(self):
        if self._ep == 0.0:
            for i, j in combinations(xrange(len(self.positions)), 2):
                r = self.positions[i] - self.positions[j]
                mic(r, self.size, self.cutoff)
                r2 = np.sum(r**2)
                if r2 <= self.cutoff2:
                    ir6 = 1.0 / r2**3
                    self._ep += 4.0 * ir6 * (ir6 - 1) - self.ecut
        return self._ep
        
    @property
    def E(self):
        ek = self.EK
        ep = self.EP
        return ek+ep, ek, ep
        
    @property
    def VCM(self):
        return np.sum(self.velocities, axis=0) / self.nparts
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def ver1(self):
        self.positions += self.velocities * self.dt + self.forces * (self.dt*self.dt * 0.5)
        self.positions = self.positions - np.rint(self.positions/self.size) * self.size
        self.velocities += 0.5 * self.forces * self.dt
        self.forces = np.zeros((self.nparts,self.D))
                
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def ver2(self):
        self.velocities += 0.5 * self.forces * self.dt
                
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_forces(self):
        cdef np.ndarray[DTYPE_t] r, f
        cdef np.float64_t r2, ff, ir2, ir6
        cdef Py_ssize_t i, j, k
        for i, j in combinations(xrange(self.nparts), 2):
                r = self.positions[i] - self.positions[j]
                mic(r, self.size, self.cutoff)
                #~ r2 = np.sum(r**2)
                r2 = 0.0
                for k in xrange(self.D):
                    r2 += r[k] * r[k]
                if r2 <= self.cutoff2:
                    f = LJ(r2) * r
                    self.forces[i] += f
                    self.forces[j] -= f
        
    def do_step(self):
        self._start_acums()
        self.ver1()
        self.update_forces()
        self.ver2()
        
    def run(self, steps=10):
        for i in xrange(steps):
            print '\r', i,
            sys.stdout.flush()
            self.do_step()
                

    
