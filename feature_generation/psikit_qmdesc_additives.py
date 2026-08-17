#!/usr/bin/env python3
"""Generate additive Psikit descriptors; special unsupported additives are skipped."""
try:
    from .psikit_qmdesc import run
except ImportError:
    from psikit_qmdesc import run

def main(): run(default_column="Additive",force_round1=True,skip_special=True)
if __name__=="__main__": main()
