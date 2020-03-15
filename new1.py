import IPython.nbformat.current as nbf
nb = nbf.read(open('new.py', 'r'), 'py')
nbf.write(nb, open('new.pynb', 'w'), 'ipynb')