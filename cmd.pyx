#! /usr/bin/env python

import sys
import numpy as np
cimport numpy as np
from itertools import combinations, product

cdef inline double LJ(double r2):
    cdef double ir2 = 1.0 / r2
    cdef double ir6 = 1.0 / r2**3
    return 48 * ir2 * ir6 * (ir6 - 0.5)
    
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
    
cdef inline mic(r, double L):
    mic = np.abs(r) > L / 2.0
    r[mic] = r[mic] - L * np.sign(r)[mic]

cdef class MD(object):
    cdef double size
    cdef int D
    cdef int nparts
    cdef double cutoff, cutoff2, ecut, dt, _e, _ek, _ep, _temp
    cdef np.ndarray positions, velocities, forces
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
    
    def _start_acums(self):
        self._e = 0.0
        self._ek = 0.0
        self._ep = 0.0
        self._temp = 0.0
        
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
                mic(r, self.size)
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
        
    def check_distances(self):
        min = self.size
        for i, j in combinations(xrange(len(self.positions)), 2):
            r = self.positions[i] - self.positions[j]
            r = r - self.size * np.floor_divide(r, self.size)
            r = np.sum(r**2)**0.5
            if r < min:
                min = r
        print 'min=', min
    
    def do_step(self):
        cdef np.ndarray r
        cdef double r2, ff
        self._start_acums()
        self.positions += self.velocities * self.dt + 0.5 * self.forces * self.dt**2
        self.positions = self.positions - np.rint(self.positions/self.size) * self.size
        self.velocities += 0.5 * self.forces * self.dt
        self.forces = np.zeros((self.nparts,self.D))
        for i, j in combinations(xrange(len(self.positions)), 2):
            r = self.positions[i] - self.positions[j]
            mic(r, self.size)
            r2 = np.sum(r**2)
            if r2 <= self.cutoff2:
                ff = LJ(r2)
                self.forces[i] += ff * r
                self.forces[j] -= ff * r
        self.velocities += 0.5 * self.forces * self.dt
        
    def run(self, steps=10):
        with open('data/Pos.dat', 'w') as df:
            for i in xrange(steps):
                print '\r', i,
                sys.stdout.flush()
                self.do_step()
                #~ for j in xrange(self.nparts):
                    #~ x, y = self.positions[j]
                    #~ vx, vy = self.velocities[j]
                    #~ df.write('%i %f %f %f %f %f %f\n'%(j+1, x, y, 0.0, vx, vy, 0.0))
                #~ df.write('\n')
            
    

    
