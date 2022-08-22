from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
exec(open("ground.py").read())
# mine
import hamiltonian
import diffeo
import sde
from utility import *
import glob, os
#
# all data defined in utility (exp2,...)
#
def run(dict,plot=False):
    import os.path
    if 'fname' in dict:
        filename=dict['fname']
    else:
        print("No filename given")
        exit(1)
    print("filename :",filename," ",dict['ext'])
    G=hamiltonian.GaussGreen(dict['ell'],0)
    no_steps=dict['no_steps']
    if isinstance(no_steps, list):
        ODE=diffeo.MultiShoot(G)
    else:
        ODE =diffeo.Shoot(G)  # use single shooting
    ODE.set_no_steps(no_steps)
    ODE.set_landmarks(dict['landmarks_n'])
    P0, v = ODE.solve()
    #print(v)
    if plot:
        plt.figure(1)
        plot_setup()
        plt.axis('equal')
        plot_RT(dict['landmarks_n'])
        plt.savefig('rt'+str(np.random.randint(1000))+'.pdf',bbox_inches='tight')
        #plot_RT(dict['landmarks'],shadow=2)
        ODE.plot_warp()
        plt.savefig(filename+str(np.random.randint(1000))+'.pdf',bbox_inches='tight')
        print("...finished.")
    #
    return v
####################################################################

def myset(i):
    include_multipleshoot=True
    include_shoot=True
    no_steps=[5,5]
    if i==1:
        def exp (x): return exp1(x)
        scale=1
    if i==2:
        def exp (x): return exp2(x)
        scale=0.1
    if i==4:
        include_shoot=False
        def exp (x): return exp4(x)
        scale=0.1
    if i==5:
        def exp (x): return exp5(x)
        scale=0.1
    #
    noise_vars=np.array([0.0, 0.01, 0.015, 0.02])
    exts=['xa', 'xb', 'xc', 'xd']
    if include_shoot:
        for i in range(noise_vars.shape[0]):
            print("===============================\nLoading data")
            dict=exp( scale*noise_vars[i] )
            dict['ext']=exts[i]
            dict['no_steps']=int(np.prod(no_steps))
            run(dict)
    if include_multipleshoot:
        exts=['xms_a', 'xms_b', 'xms_c', 'xms_d']
        for i in range(noise_vars.shape[0]):
            print("===============================\nLoading data")
            dict=exp( scale*noise_vars[i] )
            dict['ext']=exts[i]
            dict['no_steps']=no_steps
            run(dict)

def testpot():
    include_shoot=True
    scale=0.5
    Qr = get_data('data/shell1.txt')
    Qt = get_data('data/shell3.txt')
    X=np.empty((2,Qr.shape[0],Qr.shape[1]))
    X[0,:,:]=Qr; X[1,:,:]=Qt;
    dict={}
    dict['landmarks'] = X
    dict['landmarks_n'] = X
    #dict['landmarks'] = procrust1(X)
    # Set parameters
    dict['ell']=0.5; # Green's
    dict['no_steps']=[5,4]
    dict['lam']=0.0
    dict['beta']=0
    dict['fname']="figs/pot_"
    noise_vars=np.array([0.0])
    exts=['xa']
    dict['ext']=exts[0]
    run(dict,True)

    return dict

def run_rw():
    include_shoot=True
    scale=0.1
    dict={}
    #dict['landmarks'] = procrust1(X)
    # Set parameters
    dict['ell']=0.1; # Green's
    dict['no_steps']=[5,4]
    dict['lam']=0.0
    dict['beta']=0
    dict['fname']="figs/pot_"
    noise_vars=np.array([0.0])
    exts=['xa']
    dict['ext']=exts[0]
    dists = np.zeros((40,100,100))
    for i in range(40):
        print(i)
        for j in range(99):
            for k in range(j+1,100): 
                Qr = get_data('data/rw_'+str(i)+'_'+str(j)+'.txt',step=1)
                Qt = get_data('data/rw_'+str(i)+'_'+str(k)+'.txt',step=1)
                X=np.empty((2,Qr.shape[0],Qr.shape[1]))
                X[0,:,:]=Qr; X[1,:,:]=Qt;
                dict['landmarks'] = X
                dict['landmarks_n'] = X
                v = run(dict)
                dists[i,j,k] = v

    np.save('rw_dists_reg.txt',dists)
    return dict

def run_test():
    include_shoot=True
    scale=0.1
    dict={}
    #dict['landmarks'] = procrust1(X)
    # Set parameters
    dict['ell']=0.1; # Green's
    dict['no_steps']=5
    dict['lam']=0.0
    dict['beta']=0
    dict['fname']="figs/pot_"
    noise_vars=np.array([0.0])
    exts=['xa']
    dict['ext']=exts[0]
    listing = glob.glob('data/pot_*')
    dists = np.zeros((len(listing),len(listing)))
    f = open('finalOrder.txt','w')
    f.write(str(listing))
    f.close()

    for i in range(len(listing)):
        for j in range(len(listing)): 
        #for j in range(i+1,len(listing)): 
                Qr = get_data(listing[i],step=1)
                Qt = get_data(listing[j],step=1)
                X=np.empty((2,Qr.shape[0],Qr.shape[1]))
                X[0,:,:]=Qr; X[1,:,:]=Qt;
                dict['landmarks'] = X
                dict['landmarks_n'] = X
                #dict['fname'] = str(listing[i]+listing[j])
                #v = run(dict,plot=True)
                v = run(dict,plot=False)
                dists[i,j] = v

    print(dists)
    np.save('test_dists_reg.txt',dists)

    return dict

def run_final():
    include_shoot=True
    scale=0.1
    dict={}
    #dict['landmarks'] = procrust1(X)
    # Set parameters
    dict['ell']=0.1; # Green's
    dict['no_steps']=[5,4]
    dict['lam']=0.0
    dict['beta']=0
    dict['fname']="figs/pot_"
    noise_vars=np.array([0.0])
    exts=['xa']
    dict['ext']=exts[0]
    listing = glob.glob('data/pot_*')
    dists = np.zeros((len(listing),len(listing)))
    f = open('finalOrder.txt','w')
    f.write(str(listing))
    f.close()

    for i in range(len(listing)):
        for j in range(i+1,len(listing)): 
                Qr = get_data(listing[i],step=3)
                Qt = get_data(listing[j],step=3)
                X=np.empty((2,Qr.shape[0],Qr.shape[1]))
                X[0,:,:]=Qr; X[1,:,:]=Qt;
                dict['landmarks'] = X
                dict['landmarks_n'] = X
                v = run(dict)
                dists[i,j] = v

    np.save('final_dists_reg.txt',dists)

    return dict

def run_quick():
    include_shoot=True
    scale=0.1
    dict={}
    #dict['landmarks'] = procrust1(X)
    # Set parameters
    dict['ell']=0.1; # Green's
    dict['no_steps']=[5,4]
    dict['lam']=0.0
    dict['beta']=0
    dict['fname']="figs/shells/shell_"
    noise_vars=np.array([0.0])
    exts=['xa']
    dict['ext']=exts[0]
    listing = glob.glob('data/shell*')
    dists = np.zeros((len(listing),len(listing)))
    f = open('quickOrderShell.txt','w')
    f.write(str(listing))
    f.close()

    for i in range(len(listing)):
        for j in range(i+1,len(listing)): 
                Qr = get_data(listing[i],step=10)
                Qt = get_data(listing[j],step=10)
                X=np.empty((2,Qr.shape[0],Qr.shape[1]))
                X[0,:,:]=Qr; X[1,:,:]=Qt;
                dict['landmarks'] = X
                dict['landmarks_n'] = X
                v = run(dict)
                dists[i,j] = v

    np.save('quick_dists_shells.txt',dists)

    return dict

def run_mod():
    include_shoot=True
    scale=0.1
    dict={}
    #dict['landmarks'] = procrust1(X)
    # Set parameters
    dict['ell']=0.1; # Green's
    dict['no_steps']=[5,4]
    dict['lam']=0.0
    dict['beta']=0
    dict['fname']="figs/pot_"
    noise_vars=np.array([0.0])
    exts=['xa']
    dict['ext']=exts[0]
    listing = glob.glob('data/pot_*')
    dists = np.zeros((len(listing),len(listing)))
    f = open('modOrder.txt','w')
    f.write(str(listing))
    f.close()

    for i in range(len(listing)):
        for j in range(i+1,len(listing)): 
                Qr = get_data(listing[i],step=6)
                Qt = get_data(listing[j],step=6)
                X=np.empty((2,Qr.shape[0],Qr.shape[1]))
                X[0,:,:]=Qr; X[1,:,:]=Qt;
                dict['landmarks'] = X
                dict['landmarks_n'] = X
                v = run(dict)
                dists[i,j] = v

    np.save('mod_dists_reg.txt',dists)

    return dict

def run_extra():
    include_shoot=True
    scale=0.1
    dict={}
    #dict['landmarks'] = procrust1(X)
    # Set parameters
    dict['ell']=0.1; # Green's
    dict['no_steps']=[5,4]
    dict['lam']=0.0
    dict['beta']=0
    dict['fname']="figs/pot_"
    noise_vars=np.array([0.0])
    exts=['xa']
    dict['ext']=exts[0]
    listing = glob.glob('data/pot_*')
    dists = np.zeros((len(listing),len(listing)))
    f = open('extraOrder.txt','w')
    f.write(str(listing))
    f.close()
    extra = ['data/pot_276_10_10_oinochoe_1_721.txt','data/pot_211_8_8_pelike_1_700.txt','data/pot_635_22_22_cup_c_1_720.txt']

    for i in range(len(extra)):
        for j in range(len(listing)): 
                Qr = get_data(extra[i],step=3)
                Qt = get_data(listing[j],step=3)
                X=np.empty((2,Qr.shape[0],Qr.shape[1]))
                X[0,:,:]=Qr; X[1,:,:]=Qt;
                dict['landmarks'] = X
                dict['landmarks_n'] = X
                v = run(dict)
                dists[i,j] = v

    np.save('extra_dists_reg.txt',dists)

    return dict

if __name__ == "__main__":
    # do this
    plt.ion()
    #run_quick()
    #run_mod()
    #run_test()
    #run_final()
    #run_extra()
    #run_rw()
    testpot()
    #myset(1)
    #myset(4)
    #myset(5)
    #myset(2)
