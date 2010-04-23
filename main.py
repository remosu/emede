#! /usr/bin/env python

import sys

from c_md import MD

def main(steps = 10, **kargs):
    md = MD(**kargs)
    md.run(steps)


if __name__ == '__main__':
    args = {'steps':10}
    for arg in sys.argv[1:]:
        key, value = arg.split(':')
        args[key] = float(value)
    #print '#', args
    steps = int(args['steps'])
    del args['steps']
    main(steps, **args)
