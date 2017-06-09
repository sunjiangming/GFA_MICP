import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#from sklearn.linear_model import LinearRegression
#from sklearn.isotonic import IsotonicRegression
#from sklearn.metrics.pairwise import paired_distances
import os
from sklearn.cross_validation import StratifiedKFold
import numpy.ma as ma
import sys, getopt

def zeroClassFirst(X,y):
	if (y[0]>0.5):
		for i,yLabel in enumerate(y[1:]):
			if (y[i]<0.5):
				yTmp = y[i]; XTmp = X[i]
				y[i] = y[0]; X[i] = X[0]
				y[0] = yTmp; X[0] = XTmp
				break
def computeICPValues(X_properTrain,y_properTrain,X_Test,y_Test,X_calibration,y_calibration):
	# Compute nonconformity scores
	# The reason for doing this is that libsvm always uses the label of the first training 
	# example to define the negative side of the decision boundary, unless class labels are -1 and 1.
	#zeroClassFirst(X_properTrain,y_properTrain)
	# Build model a calculate nonconformity scores for the calibration set. 
	y_calibrationAlphas = X_calibration
	conditionZero = ma.masked_less_equal(y_calibration, 0.5)
	conditionOne = ma.masked_greater(y_calibration, 0.5)
	if (y_properTrain.max() > y_properTrain.min()): # The higher value response will have the higher decision value.
		alpha_zeros = np.extract(conditionZero.mask,y_calibrationAlphas)
		alpha_ones = np.extract(conditionOne.mask,-1.0*y_calibrationAlphas) # Negate to create a nonconformity score.	
	else: # The lower value response will have the higher decision value.
		print("At least two labels should be prepared!!!")
		sys.exit()
	
	alpha_zeros.sort()
	alpha_ones.sort()
	
	# Compute p-values for the test examples.
	y_testAlphas = X_Test
	# Searching is done from the left, thus a larger value of searchsorted is more nonconforming.
	# Indexing start at 0 this is why we set +2 rather than +1.
	p_zeros = 1.0-1.0*(np.searchsorted(alpha_zeros,y_testAlphas)+1)/(len(alpha_zeros)+1)
	p_ones = 1.0-1.0*(np.searchsorted(alpha_ones,-1.0*y_testAlphas)+1)/(len(alpha_ones)+1)
	
	return p_zeros,p_ones

def predictICP(X_properTrain,y_properTrain,X_Test,y_Test,X_Cal,y_Cal):
		p_0,p_1 = computeICPValues(X_properTrain,y_properTrain,X_Test,y_Test,X_Cal,y_Cal)
		return p_0, p_1

def returnConMatrix(p_0,p_1,y_t,alpha):
	tp=0
	fp=0
	tn=0
	fn=0
	uncertains=0
	emptys=0
	tp = np.count_nonzero(np.logical_and(np.logical_and(p_1>alpha, p_0<=alpha),y_t==1))
	fp = np.count_nonzero(np.logical_and(np.logical_and(p_1>alpha, p_0<=alpha),y_t==0))
	tn = np.count_nonzero(np.logical_and(np.logical_and(p_0>alpha, p_1<=alpha),y_t==0))
	fn = np.count_nonzero(np.logical_and(np.logical_and(p_0>alpha, p_1<=alpha),y_t==1))
	uncertains = np.count_nonzero(np.logical_and(p_0>alpha, p_1>alpha))
	emptys = np.count_nonzero(np.logical_and(p_0<=alpha, p_1<=alpha))
	
	error0 = (np.count_nonzero(np.logical_and( p_0<=alpha, y_t==0)))
	error1 = (np.count_nonzero(np.logical_and( p_1<=alpha, y_t==1)))
	validity0 = (error0 + 0.0)/max(sum(y_t==0),1.0)
	validity1 = (error1 + 0.0)/max(sum(y_t==1),1.0)
	validity = (error0 + error1 + 0.0)/y_t.shape[0]
	efficiency = (tp+tn+fp+fn+0.0)/y_t.shape[0]
	return tp, fp, tn, fn, uncertains, emptys,1.0-validity,1.0-validity0,1.0-validity1,efficiency


def main(argv):
	#os.chdir("/home/kjtm282/codes/gdsc/cal_test/")
	propertrainfile = '' #'both_dim_train_pIC50_fold_1_.mtx' # traing set
	testfile = '' #'fold1_GE_k100-sample-400-predictions.csv' # test set
	calfile = '' #'both_dim_cal_pIC50_fold_1_.mtx' # calibration set
	outputdir = '' # output directory
	
	try:
		opts, args = getopt.getopt(argv,"hp:t:c:o:",["propertrainfile=","testfile=","calfile=","outputdir="])
	except getopt.GetoptError:
		print("mccp_svm_openmp_sklearn.it4i.py -p <propertrainfile> -t <testfile> -c <calfile> -o <outputdir>")
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
		 print("mccp_svm_openmp_sklearn.it4i.py -p <propertrainfile> -t <testfile> -c <calfile> -o <outputdir>")
		 sys.exit()
		elif opt in ("-p", "--propertrainfile"):
			propertrainfile = arg
		elif opt in ("-t", "--testfile"):
			testfile = arg
		elif opt in ("-c", "--calfile"):
			calfile = arg
		elif opt in ("-o", "--outputdir"):
			outputdir = arg
	print(propertrainfile)
	print(testfile)
	print(calfile)
	print(outputdir)
	#create output dir
	if not os.path.exists(outputdir):
		os.makedirs(outputdir)
	#set cal part as missing value when train GFA
	# path to calibration, test and traing sets
	df_cal = pd.read_csv(calfile,sep=",")
	df_test = pd.read_csv(testfile,sep=",").as_matrix()
	df_train = pd.read_csv(propertrainfile,sep=" ")

	#train
	df_properTrain=df_train[pd.merge(df_train, df_cal, how='left',on=['row','col']).isnull().any(axis=1)]
	y_properTrain=1.0*(df_properTrain['y']>=5).values.reshape(-1,1)
	X_properTrain=df_properTrain['y'].values.reshape(-1,1)
	#test
	y_Test=1.0*(df_test[:,2]>=5)
	X_Test=df_test[:,3].reshape(-1,1)
	#calibration
	y_Cal=1.0*(df_cal['y']>=5).values.reshape(-1,1)
	X_Cal=df_cal['y'].values.reshape(-1,1)

	p_0_ICP, p_1_ICP = predictICP(X_properTrain,y_properTrain,X_Test,y_Test,X_Cal,y_Cal)

	output=np.hstack((df_test[:,0:4], p_0_ICP, p_1_ICP))
	fheader="Cell\tDrug\ty\ty_pred\tp_0_ICP\tp_0_ICP"
	np.savetxt(outputdir+'/GFA_MICP_pred_rst.txt',output,fmt='%0.3f', delimiter='\t',header=fheader)

	fdr=-999*np.ones(100)
	k=-999*np.ones(100)
	g=-999*np.ones(100)
	val = -999*np.ones(100)
	val0 = -999*np.ones(100)
	val1 = -999*np.ones(100)
	efficiency = -999*np.ones(100)
	i=0
	alpha_ranges=np.linspace(0.0001,0.5,100)
	for alpha in alpha_ranges:
		tp, fp, tn, fn, uncertains, emptys, val[i], val0[i], val1[i],efficiency[i] = returnConMatrix(p_0_ICP,p_1_ICP,y_Test.reshape(-1,1),alpha)
		sn = 1.0*tp/max(fn+tp,1)
		sp = 1.0*tn/max(fp+tn,1)
		acc = ((tp+tn)/max(tp+tn+fp+fn,1.0))
		if (tp+tn+fp+fn+0.0) > 0 :
			fdr[i] = (fp/max(fp+tp,1.0))
			cp=( (fp+tp+0.0)*(tp+fn) + (fn+tn)*(fp+tn+0.0) )/((tp+tn+fp+fn+0.0)**2)
			k[i] = (acc-cp)/(1.0-cp)
			g[i] = 1.0*(sn*sp)**0.5
		i+=1
	metrics=np.vstack((alpha_ranges,k,g,val,val0,val1,efficiency)).T
	fheader="Alpha\tKappa\tG-mean\tValidity_all\ttValidity_neg\ttValidity_pos\tefficiency"
	np.savetxt(outputdir+'/GFA_MICP_metrics.txt',metrics,fmt='%0.3f', delimiter='\t',header=fheader)
		
	alpha_ranges=np.linspace(0.0001,0.5,100)
	plt.plot(alpha_ranges, alpha_ranges, label='Diagonal line')
	plt.plot(alpha_ranges, 1.0-val1,'-*',label='Active')
	plt.plot(alpha_ranges, 1.0-val0,'-.',label='Inactive')
	plt.plot(alpha_ranges, 1.0-val,'-o',label='Both Active and Inactive')
	plt.xlabel("expected error")
	plt.ylabel("observed error")
	plt.legend( loc='upper left', numpoints = 1 )
	axes = plt.gca()
	axes.set_xlim([0,0.5])
	axes.set_ylim([0,0.5])
	plt.show()

	alpha_ranges=np.linspace(0.0001,0.5,100)
	plt.plot(alpha_ranges, efficiency,'-*',label='Efficiency')
	plt.plot(alpha_ranges, k,'-.',label='Kappa')
	plt.plot(alpha_ranges, g,'-o',label='G-mean')
	plt.xlabel("expected error")
	plt.ylabel("observed error")
	plt.legend( loc='lower right', numpoints = 1 )
	axes = plt.gca()
	axes.set_xlim([0,0.5])
	axes.set_ylim([0,1.0])
	plt.show()
	
	alpha_ranges=np.linspace(0.0001,0.5,100)
#plt.plot(alpha_ranges, eff_chem,'-o',label='Chemical features')
#plt.plot(alpha_ranges, eff_genom,'-*',label='Genomic features')
#plt.plot(alpha_ranges, eff_all,'-.',label='Both')
#plt.xlabel("expected error")
#plt.ylabel("Efficiency")
#plt.legend( loc='lower right', numpoints = 1 )
#axes = plt.gca()
#axes.set_xlim([0,0.5])
#axes.set_ylim([0,1.0])
#plt.show()

#alpha_ranges=np.linspace(0.0001,0.5,100)
#plt.plot(alpha_ranges, k_chem,'-o',label='Chemical features')
#plt.plot(alpha_ranges, k_genom,'-*',label='Genomic features')
#plt.plot(alpha_ranges, k_all,'-.',label='Both')
#plt.xlabel("expected error")
#plt.ylabel("Kappa")
#plt.legend( loc='lower right', numpoints = 1 )
#axes = plt.gca()
#axes.set_xlim([0,0.5])
#axes.set_ylim([0,1.0])
#plt.show()

#alpha_ranges=np.linspace(0.0001,0.5,100)
#plt.plot(alpha_ranges, g_chem,'-o',label='Chemical features')
#plt.plot(alpha_ranges, g_genom,'-*',label='Genomic features')
#plt.plot(alpha_ranges, g_all,'-.',label='Both')
#plt.xlabel("expected error")
#plt.ylabel("G-mean")
#plt.legend( loc='lower right', numpoints = 1 )
#axes = plt.gca()
#axes.set_xlim([0,0.5])
#axes.set_ylim([0,1.0])
#plt.show()
