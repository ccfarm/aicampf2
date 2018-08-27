import os
def readlog(logpath,lastlen):
    logdata = []
    loglen = 0

    if not os.path.isfile(logpath):
        return loglen,logdata
    logfile=open(logpath,'r')

    # logfile current size
    loglen=os.path.getsize(logfile)

    # find the zengliang data of logfile
    logfile.seek(lastlen,0)
    logdata=logfile.readlines()

    # you need get loglen as the next input of lastlen
    return loglen,logdata
