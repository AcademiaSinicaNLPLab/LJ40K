from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from subprocess import call

def run(eid):
    
    train_data_root = '/home/doug919/projects/data/MKLv2/2000samples_4/train'
    test_data_root = '/home/doug919/projects/data/MKLv2/2000samples_4/test'
    train_data_tag = '800p800n_Xy'
    test_data_tag = '200p200n_Xy'
    output_prefix = 'Thread%d_E1_800' % (eid)

    cmd = 'matlab -r "mklv2_exp_2(%d, \'%s\', {\'TFIDF\', \'keyword\'}, \'%s\', \'%s\', \'%s\', \'%s\');exit;" > log/log_thread_%d' % \
        (eid, output_prefix, train_data_root, test_data_root, train_data_tag, test_data_tag, eid)

    print '> run:',cmd
    call(cmd, shell=True)

if __name__ == "__main__":

    eids = range(1, 21) 
    pool = ThreadPool(len(eids))
    res = pool.map(run, eids)
    pool.close()
    pool.join()

