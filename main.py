from FuxiRec import FuxiRec
from util.conf import ModelConf

if __name__ == '__main__':
    # Register your model here
    classic_baselines = ['c1. MF', 'c2. NeuMF']
    graph_baselines = ['g1. LightGCN', 'g2. LinkProp']
    graph_signal = ['gs1. ChebyCF',]
    negative_sampling = ['ns1. MixGCF', 'ns2. AHNS']
    ssl_graph_models = ['sg1. SGCL']
    ssl_sequential_models = ['ss1. CL4SRec','ss2. DuoRec','ss3. BERT4Rec']
    autoencoder_baselines = ['a1. CDAE', 'a2. MultiVAE']
    diffusion_models = ['d1. CODIGEM', 'd2. DiffRec', 'd3. BSPM', 'd4. GiffCF', 'd5. DDRM', 'd6. HDRM', 'd7. TV-Diff',]
    
    print('=' * 80)
    print('   FuxiRec: A library for general recommendation.   ')
    print('=' * 80)
    
    print('Classic Baseline Models:')
    print('   '.join(classic_baselines))
    print('-' * 100)
    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Graph Signal Processing Models:')
    print('   '.join(graph_signal))
    print('-' * 100)
    print('Negative Sampling Models:')
    print('   '.join(negative_sampling))
    print('-' * 100)
    print('Self-Supervised Graph-Based Models:')
    for i in range(len(ssl_graph_models)//7 + 1):   
        print('   '.join(ssl_graph_models[i*7:(i+1)*7]))
    print('=' * 80)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('-' * 100)
    print('Autoencoder-Based Models:')
    print('   '.join(autoencoder_baselines))
    print('=' * 100)
    print('Diffusion Models:')
    print('   '.join(diffusion_models))
    print('-' * 80)
    model = input('Please enter the model you want to run:').lower()
    import time

    s = time.time()
    code2model = {'c1':'MF', 'c2':'NeuMF',
                  'g1':'LightGCN', 'g2':'LinkProp',
                  'gs1': 'ChebyCF',
                  'ns1':'MixGCF', 'ns2':'AHNS',
                  'sg1': 'SGCL',
                  'ss1':'CL4SRec', 'ss2':'DuoRec', 'ss3':'BERT4Rec',
                  'a1':'CDAE', 'a2':'MultiVAE',
                  'd1': 'CODIGEM', 'd2': 'DiffRec', 'd3': 'BSPM', 'd4': 'GiffCF', 'd5': 'DDRM', 'd6': 'HDRM', 'd7':'TV_Diff'}
    try:
        conf = ModelConf('./conf/' + code2model[model] + '.conf')
    except:
        print('Wrong model name!')
        exit(-1)
    rec = FuxiRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
