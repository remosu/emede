#! /usr/bin/env python

import sys

from cmd import MD


if __name__ == '__main__':
    args = {'steps':10}
    for arg in sys.argv[1:]:
        key, value = arg.split(':')
        args[key] = float(value)
    #print '#', args
    steps = int(args['steps'])
    del args['steps']
    m = MD(**args)
    m.run(steps)
