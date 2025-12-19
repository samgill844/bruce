def main():
    from astropy.time import Time 
    import sys
    print(Time(float(sys.argv[1])+2457000, format='jd').datetime)