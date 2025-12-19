def main():
    from astropy.time import Time 
    import sys
    print(Time(sys.argv[1], format='jd').datetime)