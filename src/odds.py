def kelly_fraction(p,o):
    b=o-1.0
    return max(0.0,(b*p-(1-p))/b) if b>0 else 0.0
