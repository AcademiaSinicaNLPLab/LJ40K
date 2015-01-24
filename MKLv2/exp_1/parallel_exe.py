from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from subprocess import call

def run(eid):
    
    #cmd = 'matlab -r "mkl(%d)"' % (eid)
    output_prefix = 'Thread%d_E1' % (eid)
    cmd = 'matlab -r "mklv2_exp_1(%d, \'%s\', {\'keyword\', \'image_rgba_gist\', \'image_rgba_phog\', \'TFIDF\'});exit;" > log/log_tread_%d' % (eid, output_prefix, eid)

    print '> run:',cmd
    call(cmd, shell=True)

if __name__ == "__main__":

    eids = range(1, 41) 
    pool = ThreadPool(len(eids))
    res = pool.map(run, eids)
    pool.close()
    pool.join()

