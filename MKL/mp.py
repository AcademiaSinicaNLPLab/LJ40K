from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from subprocess import call

def run(eid):
    
    #cmd = 'matlab -r "mkl(%d)"' % (eid)
    cmd = 'matlab -r "mklTest(%d)"' % (eid)

    print '> run:',cmd
    call(cmd, shell=True)

if __name__ == "__main__":

    eids = range(41) 
    pool = ThreadPool(len(eids))
    res = pool.map(run, eids)
    pool.close()
    pool.join()

