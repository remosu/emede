#!/usr/bin/env python

import pstats, cProfile

import pyximport
pyximport.install()

import c_md

m = c_md.MD(D=2.1, nparts=64.0, dt=0.001)
cProfile.runctx("m.run(steps=100)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
