def normalize_histogram(bins,w):
    for i in range(len(bins)):
        bins[i]=float(bins[i])
    total=sum(bins)
    if total==0:
        return [0.0]*len(bins)
    for i in range(0,len(bins)):
        bins[i]=float(bins[i])*w/float(total)
#     print bins
    return bins

def normalize_histogram_sqr(bins):
    total=0
    for bin in bins:
        total+=bin**2

    if total==0:
        return [0.0]*len(bins)
    for i in range(0,len(bins)):
        if bins[i]<0:
            bins[i]=-float(bins[i])**2/float(total)
        else:
            bins[i]=float(bins[i])**2/float(total)
#     print bins
    return bins
def normalize_histogram_abs(bins,w):
    for i in range(len(bins)):
        bins[i]=float(bins[i])
    total=0
    for bin in bins:
        total+=abs(bin)

    if total==0:
        return [0.0]*len(bins)
    for i in range(0,len(bins)):
        bins[i]=float(bins[i])*w/float(total)
    return bins
