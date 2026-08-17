#!/usr/bin/env python3
"""Generate solvent Psikit descriptors without overwriting existing output."""
try:
    from .psikit_qmdesc import run
except ImportError:
    from psikit_qmdesc import run

def main(): run(default_column="Solvent")
if __name__=="__main__": main()
