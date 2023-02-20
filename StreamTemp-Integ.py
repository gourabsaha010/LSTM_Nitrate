import sys
sys.path.append('../') #('C://Users//fzr5082//Desktop//hydroDL-dev-master//hydroDL-dev-master')   #('../')
from hydroDL import master, utils
from hydroDL.master import default
#from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat
from hydroDL.utils import time

import numpy as np
import os
import torch
import pandas as pd
import math
import random

forcing_list = [
                '3_Frcgs7_9sites_WQS0001_grp2_trn_gpfl.feather',
                '3_Frcgs7_9sites_WQS0001_grp2_trn_gpfl.feather',
                '3_Frcgs7_9sites_WQS0001_grp2_trn_gpfl.feather',
                '3_Frcgs7_9sites_WQS0001_grp2_trn_gpfl.feather'
                ]

attr_list = [
            '3_AttributeData2_9Sites_WQS0001_grp2_trn_gpfl.feather',
            '3_AttributeData2_9Sites_WQS0001_grp2_trn_gpfl.feather',
            '3_AttributeData2_9Sites_WQS0001_grp2_trn_gpfl.feather',
            '3_AttributeData2_9Sites_WQS0001_grp2_trn_gpfl.feather'
             ]

# Batch_list = [41, 41, 41, 41, 41, 41, 41, 41, 41]
Batch_list = [6, 6, 6, 6]           ## 21, 21, 21, 21, 42, 42, 42, 42, 17, 17, 17, 17, 34, 34, 34, 48, 48, 48 #Batch_list
Hidden_list = [6, 10, 12, 15]          # 45, 50, 45, 50, 45, 50, 45, 50, 40, 50, 40, 50, 40, 50, 40, 50, 60, 60,  # #Hidden_list
RHO_list = [150, 150, 180, 180]         # 150, 150, 175, 175, 150, 150, 175, 175
Randomseed = [0, 1, 3, 5]     # 0, 1, 3, 5, 7, 9
#for seed in Randomseed:
for f_list, a_list, b_list, h_list, rh_list in zip(forcing_list, attr_list, Batch_list, Hidden_list, RHO_list):    #Batch_list, Hidden_list
    for seed in Randomseed:

        global TempTarget
        # Options for different interface
        interfaceOpt = 1
        # ==1 is the more interpretable version, explicitly load data, model and loss, and train the model.
        # ==0 is the "pro" version, efficiently train different models based on the defined dictionary variables.
        # the results are identical.

        # Options for training and testing
        # 0: train base model without DI
        # 0,1: do both at the same time
        # 1: train DI model
        # 2: test trained models
        Action = [0, 2]  # it was [0 , 1]
        # Set hyperparameters for training or retraining
        EPOCH = 2
        BATCH_SIZE = b_list
        RHO = rh_list
        HIDDENSIZE = h_list
        saveEPOCH = 1
        # it was 50
        Ttrain = [20120101, 20200101]  # Training period. it was [19851001, 19951001]

        #### Set hyperparameters for Pre-training the model #####
        retrained = False   # True: means you want to train a pre-trained model, False: means you want to train a new model
        freeze = False
        pre_Ttrain = [20120101, 20150101]
        pre_EPOCH = 500
        pre_BATCH_SIZE = 686
        pre_HIDDENSIZE = 100
        pre_RHO = 365
        ###############################
        TempTarget = 'Nitrate_rand_14'   # 'obs' or     or Q9Tw   '00010_Mean'  outlet_tave_water


        absRoot = os.getcwd()


        # Define root directory of database and output
        # Modify this based on your own location
        rootDatabase = os.path.join(os.path.sep, absRoot, 'scratch', 'SNTemp')  # CAMELS dataset root directory: /scratch/Camels
        rootOut = os.path.join(os.path.sep, absRoot, 'TempDemo', 'FirstRun')  # Model output root directory: /data/rnnStreamflow

        forcing_path = os.path.join(os.path.sep, rootDatabase, 'Forcing', 'Forcing_new', f_list)  # obs_18basins
        forcing_data =pd.read_feather(forcing_path)
        attr_path = os.path.join(os.path.sep, rootDatabase, 'Forcing', 'attr_new', a_list)
        attr_data = pd.read_feather(attr_path)
        camels.initcamels(forcing_data, attr_data, TempTarget, rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict
        D_N_P_path = os.path.join(os.path.sep, rootDatabase, 'Forcing', 'Forcing_new', '3_D_N_P_9sites_WQS0001_grp2_gpfl.xlsx') # D_N_P.xlsx'


        # Define all the configurations into dictionary variables
        # three purposes using these dictionaries. 1. saved as configuration logging file. 2. for future testing. 3. can also
        # be used to directly train the model when interfaceOpt == 0
        # define dataset
        optData = default.optDataCamels
        optData = default.update(optData, tRange=Ttrain, target='StreamTemp', doNorm=[True,True])  # Update the training period
        # define model and update parameters
        if torch.cuda.is_available():
            optModel = default.optLstm
        else:
            optModel = default.update(
                default.optLstm,
                name='hydroDL.model.rnn.CpuLstmModel')
        optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
        # define loss function
        optLoss = default.optLossRMSE
        # define training options
        optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH, seed=seed)
        # define output folder for model results
        exp_name = 'TempDemo'
        exp_disp = 'FirstRun'


        save_path = os.path.join(absRoot, exp_name, exp_disp, \
                    'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}_{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                                                  optTrain['miniBatch'][1],
                                                                                  optModel['hiddenSize'],
                                                                                  optData['tRange'][0], optData['tRange'][1],seed))
        out = os.path.join(rootOut, save_path, 'All-2010-2016') # output folder to save results

        ##############################################################
        # save path and out for saving results for when had some pretraining  ####
        pre_save_path = os.path.join(absRoot, exp_name, exp_disp, \
                                                 'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(
                                                     pre_EPOCH,
                                                     pre_BATCH_SIZE,
                                                     pre_RHO,
                                                     pre_HIDDENSIZE,
                                                     pre_Ttrain[0],
                                                     pre_Ttrain[1]))
        pre_out = os.path.join(rootOut, pre_save_path, 'All-2010-2016')
        if retrained == True:          # the path for retrained model
            out_retrained = os.path.join(rootOut, 'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                                                  optTrain['miniBatch'][1],
                                                                                  optModel['hiddenSize'],
                                                                                  optData['tRange'][0], optData['tRange'][1]),
                                         'ST_epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(
                                             pre_EPOCH,
                                             pre_BATCH_SIZE,
                                             pre_RHO,
                                             pre_HIDDENSIZE,
                                             pre_Ttrain[0],
                                             pre_Ttrain[1]))

        ##############################################################




        # Wrap up all the training configurations to one dictionary in order to save into "out" folder
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)

        # Train the base model without data integration
        if 0 in Action:
            if interfaceOpt == 1:  # use the more interpretable version interface

                #fixing random seeds
                if optTrain['seed'] is None:
                    # generate random seed
                    randomseed = int(np.random.uniform(low=0, high=1e6))
                    optTrain['seed'] = randomseed
                    print('random seed updated!')
                else:
                    randomseed = optTrain['seed']

                random.seed(randomseed)
                torch.manual_seed(randomseed)
                np.random.seed(randomseed)
                torch.cuda.manual_seed(randomseed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False



                # load data
                df, x, y, c = master.loadData(optData, TempTarget, forcing_path, attr_path, out)  # df: CAMELS dataframe; x: forcings; y: streamflow obs; c:attributes
                # main outputs of this step are numpy ndArrays: x[nb,nt,nx], y[nb,nt, ny], c[nb,nc]
                # nb: number of basins, nt: number of time steps (in Ttrain), nx: number of time-dependent forcing variables
                # ny: number of target variables, nc: number of constant attributes
                nx = x.shape[-1] + c.shape[-1]  # update nx, nx = nx + nc
                ny = y.shape[-1]
                #path = os.path.join('G:\Farshid\CONUS_Temp\Example3\TempDemo\FirstRun\epochs2000_batch686_rho365_hiddensize100_Tstart20101001_Tend20141001\All-2010-2016\\model_Ep1500.pt')
               # model = torch.load(path)
                # load model for training
                if retrained == False:
                    if torch.cuda.is_available():
                        model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
                    else:
                        model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
                else:    # we want to train a pre_trained model
                    modelname = 'model_Ep'+str(pre_EPOCH)+'.pt'
                    path = os.path.join(os.path.sep, pre_out, modelname)
                    model = torch.load(path)
                optModel = default.update(optModel, nx=nx, ny=ny)
                # the loaded model should be consistent with the 'name' in optModel Dict above for logging purpose
                lossFun = crit.RmseLoss()
                # the loaded loss should be consistent with the 'name' in optLoss Dict above for logging purpose
                # update and write the dictionary variable to out folder for logging and future testing
                masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
                master.writeMasterFile(masterDict)
                # train model
                if retrained == False:
                    out1 = out
                else:
                    out1 = os.path.join(rootOut, pre_save_path, 'ST_epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                                                  optTrain['miniBatch'][1],
                                                                                  optModel['hiddenSize'],
                                                                                  optData['tRange'][0], optData['tRange'][1]))
                    if not os.path.exists(out1):
                        os.mkdir(out1)
                    dict = masterDict
                    dict['out'] = out_retrained
                    master.writeMasterFile(dict)
                ### for freezing several parts of model
                if retrained & freeze:
                    count = 0
                    for param in model.parameters():
                        if count == 6 or count == 7:
                            param.requires_grad = False
                        count = count + 1
                ############
                model = train.trainModel(
                    model,
                    x,
                    y,
                    c,
                    lossFun,
                    nEpoch=EPOCH,
                    miniBatch=[BATCH_SIZE, RHO],
                    saveEpoch=saveEPOCH,
                    saveFolder=out1,
                    D_N_P_path=D_N_P_path)
            elif interfaceOpt==0: # directly train the model using dictionary variable
                master.train(masterDict)


        # Train DI model
        if 1 in Action:
            nDayLst = [1, 3]
            for nDay in nDayLst:
                # nDay: previous Nth day observation to integrate
                # update parameter "daObs" for data dictionary variable
                optData = default.update(default.optDataCamels, daObs=nDay)
                # define output folder for DI models
                out = os.path.join(rootOut, save_path, 'All-85-95-DI' + str(nDay))
                masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
                if interfaceOpt==1:
                    # load data
                    df, x, y, c = master.loadData(optData)
                    # optData['daObs'] != 0, return a tuple to x, x[0]:forcings x[1]: integrated observations
                    x = np.concatenate([x[0], x[1]], axis=2)
                    nx = x.shape[-1] + c.shape[-1]
                    ny = y.shape[-1]
                    # load model for training
                    if torch.cuda.is_available():
                        model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
                    else:
                        model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
                    optModel = default.update(optModel, nx=nx, ny=ny)
                    lossFun = crit.RmseLoss()
                    # update and write dictionary variable to out folder for logging and future testing
                    masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
                    master.writeMasterFile(masterDict)
                    # train model
                    model = train.trainModel(
                        model,
                        x,
                        y,
                        c,
                        lossFun,
                        nEpoch=EPOCH,
                        miniBatch=[BATCH_SIZE, RHO],
                        saveEpoch=saveEPOCH,
                        saveFolder=out)
                elif interfaceOpt==0:
                    master.train(masterDict)

        # Test models
        if 2 in Action:
            TestEPOCH = 2   # it was 200  # choose the model to test after trained "TestEPOCH" epoches
            # generate a folder name list containing all the tested model output folders
            caseLst = ['All-2010-2016']
            nDayLst = [] #[1, 3]
            for nDay in nDayLst:
                caseLst.append('All-85-95-DI' + str(nDay))
            if retrained == True:
                outLst = [os.path.join(rootOut, out_retrained)]
            else:
                outLst = [os.path.join(rootOut, save_path, x) for x in caseLst]
            subset = 'All'  # 'All': use all the CAMELS gages to test; Or pass the gage list
            tRange = [20120101, 20200101]  # Testing period
            predLst = list()
            obsLst = list()
            predLst_res = list()
            obsLst_res = list()
            statDictLst = []
            for i, out in enumerate(outLst):
                #df, pred, obs = master.test(out, TempTarget, forcing_path[i], attr_path[i], tRange=tRange, subset=subset, basinnorm=True, epoch=TestEPOCH, reTest=True)
                df, pred, obs, x = master.test(out, TempTarget, forcing_path, attr_path, D_N_P_path, tRange=tRange, subset=subset, basinnorm=False, epoch=TestEPOCH, reTest=True)

                # change the units ft3/s to m3/s
                #obs = obs * 0.0283168
                #pred = pred * 0.0283168
                ### We are substitute the obs into NaN outside the testing period, according to D_N_P file
                D_N_P = pd.read_excel(D_N_P_path)
                for ii in range(obs.shape[0]):
                    # Testing means testing!
                    tLst1 = D_N_P.iloc[ii]['S_Training']  # 'S_Testing', 'S_Training'
                    tLst2 = D_N_P.iloc[ii]['E_Training']  # 'E_Testing', 'E_Training'
                    tArray1 = time.tRange2Array([int(str(tLst1.year) + str(tLst1.month).zfill(2) + str(tLst1.day).zfill(2)),
                                                 int(str(tLst2.year) + str(tLst2.month).zfill(2) + str(tLst2.day).zfill(2))])
                    tArray2 = time.tRange2Array(tRange)
                    C, ind1, ind2 = np.intersect1d(tArray1, tArray2, return_indices=True)
                    # obs[ii, ~ind2, :] = np.nan
                    arr = np.arange(obs.shape[1])
                    ind3 = np.delete(arr, ind2)
                    obs[ii, ind3, :] = np.nan


                #####################################################################

                predLst.append(pred) # the prediction list for all the models
                obsLst.append(obs)
                np.save(os.path.join(out, 'pred.npy'), pred)
                np.save(os.path.join(out, 'obs.npy'), obs)
                #np.save(os.path.join(out, 'obs.npy'), obs)

                f = np.load(os.path.join(out, 'x.npy'))  # it has been saved previously in the out directory (forcings)
                T = (f[:, :, 3] + f[:, :, 4]) / 2    # mean air T for T_residual
                T_air = np.expand_dims(T, axis=2)
                pred_res = pred - T_air
                obs_res = obs - T_air
                predLst_res.append(pred_res)
                obsLst_res.append(obs_res)
            # calculate statistic metrics
               # statDict = stat.statError(pred.squeeze(), obs.squeeze())
              #  statDictLst.append([statDict])
        #    statDictLst1 = [stat.statError(x.squeeze(), obs.squeeze()) for x, y in predLst]
            statDictLst = [stat.statError(x.squeeze(), y.squeeze()) for (x, y) in zip(predLst, obsLst)]
            statDictLst_res = [stat.statError_res(x.squeeze(), y.squeeze(), z.squeeze(), w.squeeze()) for (x, y, z, w) in
                           zip(predLst, obsLst, predLst_res, obsLst_res)]
            ### save this file too
            # median and STD calculation
            count = 0
            mdstd = np.zeros([len(statDictLst_res[0]),3])
            for i in statDictLst_res[0].values():
                median = np.nanmedian((i))    # abs(i)
                STD = np.nanstd((i))        # abs(i)
                mean = np.nanmean((i))      #abs(i)
                k = np.array([[median,STD, mean]])
                mdstd[count] = k
                count = count +1
            mdstd = pd.DataFrame(mdstd, index=statDictLst_res[0].keys(), columns=['median', 'STD','mean'])
            if retrained==True:
                mdstd.to_csv((os.path.join(rootOut, out_retrained, "mdstd.csv")))
            else:
                mdstd.to_csv((os.path.join(rootOut, save_path, "mdstd.csv")))

            #Show boxplots of the results
            plt.rcParams['font.size'] = 14
            keyLst = ['Bias', 'RMSE', 'ubRMSE', 'NSE', 'Corr']
            dataBox = list()
            for iS in range(len(keyLst)):
                statStr = keyLst[iS]
                temp = list()
                for k in range(len(statDictLst_res)):
                    data = statDictLst_res[k][statStr]
                    data = data[~np.isnan(data)]
                    temp.append(data)
                dataBox.append(temp)
            labelname = ['20%pub', '1%_data', '20%_data', '70%_data', '90%_data', '98%_data',
                         '100%_data']  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']
            for nDay in nDayLst:
                labelname.append('DI(' + str(nDay) + ')')
            xlabel = ['Bias ($\mathregular{deg}$C)', 'RMSE', 'ubRMSE', 'NSE', 'Corr']
            fig = plot.plotBoxFig(dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8))
            fig.patch.set_facecolor('white')
            boxPlotName = "Target:" + TempTarget + " ,epochs=" + str(TestEPOCH) + " ,Hiddensize=" + str(
                HIDDENSIZE) + " ,RHO=" + str(RHO) + " ,Batches=" + str(BATCH_SIZE)
            fig.suptitle(boxPlotName, fontsize=12)
            plt.rcParams['font.size'] = 12

            plt.savefig(os.path.join(rootOut, save_path, (str(len(forcing_path)) + "Boxplot.png")))  # , dpi=500
            #fig.show()

            # Plot timeseries and locations

            gageindex = [0]
            t = utils.time.tRange2Array(tRange)

            attr = pd.read_feather(attr_path)

            # plot.TempSeries_4_Plots_ERL(attr_path, statDictLst_res, obs, predLst, TempTarget, tRange, boxPlotName, rootOut, save_path, sites=18, Stations=None)

            plot.plotMap(statDictLst[0]['NSE'], lat=attr['lat'].to_numpy(), lon=attr['lon'].to_numpy(),
                         title='NSE' + boxPlotName)

            plt.savefig((os.path.join(rootOut, save_path, "MapNSE-LowRes.png")), bbox_inches='tight')
            #plt.show()
