import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from hydroDL import utils
import string
import random
import os
#os.environ[
 #    'PROJ_LIB'] = r'G:\Farshid\myenvs\conda-meta'
 #r'/opt/anaconda/pkgs/proj4-5.2.0-he6710b0_1/share/proj/'
from mpl_toolkits import basemap
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates
def plotBoxFig(data,
               label1=None,
               label2=None,
               #colorLst=['darkred', 'red', 'pink','darkblue', 'blue', 'deepskyblue','black', 'gray', 'lightgray'],
               #colorLst=['darkred', 'darkblue','red', 'blue','pink', 'deepskyblue','black', 'gray', 'lightgray'],
               #colorLst=['red', 'blue', 'black', 'darkred', 'darkblue','pink', 'deepskyblue', 'gray', 'lightgray'],
               colorLst=[ 'darkblue', 'blue', 'deepskyblue', 'red','black', 'darkred','pink', 'gray', 'lightgray'],
               title=None,
               figsize=(10, 8),
               sharey=True,
               xticklabel=None,
               ):
    nc = len(data)
    fig, axes = plt.subplots(ncols=nc, sharey=sharey, figsize=figsize, constrained_layout=True)
    # lowlim = [-1.5, 0.6, 0.5, 0.8, 0.9, 0.6, 0.75]
    # highlim = [0.6, 1.85, 1.8, 1.01, 1.001, 1.01, 1.01]
    # step = [0.25, 0.1, 0.2, 0.025, 0.01, 0.05, 0.05]
    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        temp = data[k]
        if type(temp) is list:
            for kk in range(len(temp)):
                tt = temp[kk]
                if tt is not None and tt != []:
                    tt = tt[~np.isnan(tt)]
                    temp[kk] = tt
                else:
                    temp[kk] = []
        else:
            temp = temp[~np.isnan(temp)]
        bp = ax.boxplot(temp, patch_artist=True, notch=True, showfliers=False, widths= 0.4)  #  , whis=[5, 95]
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=colorLst[kk])

        if label1 is not None:
            ax.set_xlabel(label1[k], fontsize=17)
        else:
            ax.set_xlabel(str(k))

        # ax.set(ylim = (lowlim[k], highlim[k]))
        # ax.set_yticks(np.arange(lowlim[k], highlim[k], step[k]))

        if xticklabel is None:
            ax.set_xticks([])
        else:
            ax.set_xticks([y+1 for y in range(0,len(data[k]),2)])
            ax.set_xticklabels(xticklabel)
        # ax.ticklabel_format(axis='y', style='sci')
    if label2 is not None:
        if nc == 1:
            ax.legend(bp['boxes'], label2, loc='center', frameon=False, ncol=1, fontsize=15)
        else:
            fig.legend(bp['boxes'], label2, loc='lower center', bbox_to_anchor=(0., 1.02, 1., .102), frameon=False, ncol=4, fontsize=20, borderaxespad=0.) #it was ax[-1].legend()
           # ax.legend(bp['boxes'], label2, loc='best', frameon=False, ncol=1, fontsize=15)
    if title is not None:
        fig.suptitle(title)
    return fig

def plotBoxF(data,
               label1=None,
               label2=None,
               colorLst='rrbbkkggccmmyy',
               title=None,
               figsize=(10, 8),
               sharey=True,
               xticklabel=None,
               ylabel=None,
               subtitles=None
               ):
    nc = len(data)
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=sharey, figsize=figsize, constrained_layout=True)
    axes = axes.flat
    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        bp = ax.boxplot(
            data[k], patch_artist=True, notch=True, showfliers=False)
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=colorLst[0])
        if k == 2:
            yrange = ax.get_ylim()
        if k == 3:
            ax.set(ylim=yrange)
        ax.axvline(len(data[k])-2+0.5, ymin=0, ymax=1, color='k',
                   linestyle='dashed', linewidth=1)
        if ylabel[k] != 'NSE':
            ax.axhline(0, xmin=0, xmax=1,color='k',
                       linestyle='dashed', linewidth=1)

        if label1 is not None:
            ax.set_xlabel(label1[k])
        if ylabel is not None:
            ax.set_ylabel(ylabel[k])
        if xticklabel is None:
            ax.set_xticks([])
        else:
            ax.set_xticks([y+1 for y in range(0,len(data[k]))])
            ax.set_xticklabels(xticklabel)
        if subtitles is not None:
            ax.set_title(subtitles[k], loc='left')
        # ax.ticklabel_format(axis='y', style='sci')
    if label2 is not None:
        if nc == 1:
            ax.legend(bp['boxes'], label2, loc='best', frameon=False, ncol=2)
        else:
            axes[-1].legend(bp['boxes'], label2, loc='best', frameon=False, ncol=2, fontsize=12)
    if title is not None:
        fig.suptitle(title)
    return fig


def TempSeries_4_Plots_Nitrate(attr_path, statDictLst_res, obs, predLst, TempTarget, tRange, boxPlotName, rootOut, save_path, sites=18, Stations=None, retrained=False):
    # fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    # axes = axes.flat
    npred = 2  # 2  # plot the first two prediction: Base LSTM and DI(1)
    #subtitle = ['(seg_id_nat:1450)', '(seg_id_nat:1566)', '(seg_id_nat:1718)', '(seg_id_nat:2013)']
    txt = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    ylabel = 'Nitrate Concentration (mg/L)'


    seg_id_nat = []
    inputdata = pd.read_feather(attr_path)
    # tRange = [20141001, 20161001]
    gage = []
    if Stations == None:
        seg_id_nat = inputdata['site_no'].unique()
    else:
        seg_id_nat = Stations
    if sites > len(seg_id_nat):
        sites = len(seg_id_nat)
    AA = random.sample(range(0, len(seg_id_nat)), sites)
    AA.sort()
    BB = [seg_id_nat[(x)] for x in AA]
    seg_id_nat = BB
    #seg_id_nat.sort()
    gage = [jj for jj in range(sites)]
    for i in range(1):
        gageindex = gage  #[i * 4:(i + 1) * 4]
        print(gageindex)
        t = utils.time.tRange2Array(tRange)
        fig, axes = plt.subplots(2, 1, figsize=(10.7, 13), constrained_layout=True)
        axes = axes.flat
        npred = len(predLst)  # 2  # plot the first two prediction: Base LSTM and DI(1)
        # subtitle = txt[i] + ' (Station ID:' + str(seg_id_nat[i]) + ') '
        # if i < (math.ceil(len(gage) / 4) - 1):
        #     subtitle = ['(a) (Station ID:' + str(seg_id_nat[4 * i]) + ') ',
        #                 '(b) (Station ID:' + str(seg_id_nat[4 * i + 1]) + ') ',
        #                 '(c) (Station ID:' + str(seg_id_nat[4 * i + 2]) + ') ',
        #                 '(d) (Station ID:' + str(seg_id_nat[4 * i + 3]) + ') ']
        # elif i == (math.ceil(len(gage) / 4) - 1):
        #     if ((len(gage)) - (i * 4)) == 1:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 2:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 3:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 2]) + ')']
        #txt = ['a', 'b', 'c', 'd']
        ylabel = 'Nitrate Concentration (mg/L)'

        for k in range(len(gageindex)):
            # iGrid = AA[gageindex[k]]
            iGrid = inputdata.index[inputdata['site_no'] == seg_id_nat[AA[gageindex[k]]]].values[0]
            yPlot = [obs[iGrid, :]]
            for y in predLst[0:npred]:
                yPlot.append(y[iGrid, :])
            # get the NSE value of LSTM and DI(1) model
            # Metrics = '[' + str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2)) +',\n'+str(np.round(statDictLst_res[0]['NSE'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr'][iGrid], 2)) + ',\n' +str(np.round(statDictLst_res[0]['NSE_res'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr_res'][iGrid], 2)) + ']'
            # subtitle1 = txt[k] + ' (Site: ' + str(seg_id_nat[k]) + ') ' + \
            subtitle1 =  ' (Site: ' + str(seg_id_nat[k]) + ') ' +\
                '[' + "RMSE: " + \
                      str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2))  +  \
                ', ' +  "NSE: " + str(np.round(statDictLst_res[0]['NSE'][iGrid], 2))  + \
                ', ' +  "Bias: " + str(np.round(statDictLst_res[0]['Bias'][iGrid], 2))  + \
                ', ' +  "Corr: " + str(np.round(statDictLst_res[0]['Corr'][iGrid], 2))  + \
                ', ' +  "KGE: " + str(np.round(statDictLst_res[0]['KGE'][iGrid], 2)) + \
                ']'
             #NSE_LSTM = [] #str(round(statDictLst[0]['NSE'][iGrid], 2))
            # NSE_DI1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
            # plot time series
            plotTS(
                t,
                yPlot,
                ax=axes[k],
                cLst='rkcbbkrmg',
                markerLst='o---+1o---+1',
                legLst=['Observed', "Hindcasted", 'Predicted_test', 'Predicted_train',
                        'Observed1',"Hindcasted1", 'Predicted_test1', 'Predicted_train1'],
                title=subtitle1, linespec=['o', '-', '-', '-', ':', '+'],  #legLst=legLst=[TempTarget, 'LSTM: ' + NSE_LSTM], title=subtitle[k]
                ylabel=ylabel , figNo=k)  # ['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DI1]
        #boxPlotName = 'Time Series simulated and observed data in testing period- values in brackets are [RMSE, NSE, Bias, NSE_res, Corr_res]'
        fig.suptitle(boxPlotName, fontsize=12.2)

        #plotName = "TempSeries.eps"
        plotName = "TempSeries.png"


        if retrained is True:
            plt.savefig(os.path.join(rootOut, out_retrained, plotName), dpi=200)
            plt.savefig(os.path.join(rootOut, out_retrained, '-LowRes'+plotName))
        else:
            plt.savefig(os.path.join(rootOut, save_path, plotName), dpi=200, bbox_inches='tight' )
            plt.savefig(os.path.join(rootOut, save_path, '-LowRes'+plotName), bbox_inches='tight' )
        fig.show()

def TempSeries_4_Plots_Nitrate2(attr_path, statDictLst_res, obs, predLst, TempTarget, tRange, boxPlotName, rootOut, save_path, sites=18, Stations=None, retrained=False):
    # fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    # axes = axes.flat
    npred = 2  # 2  # plot the first two prediction: Base LSTM and DI(1)
    #subtitle = ['(seg_id_nat:1450)', '(seg_id_nat:1566)', '(seg_id_nat:1718)', '(seg_id_nat:2013)']
    txt = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    ylabel = 'Nitrate Concentration (mg/L)'


    seg_id_nat = []
    inputdata = pd.read_feather(attr_path)
    # tRange = [20141001, 20161001]
    gage = []
    if Stations == None:
        seg_id_nat = inputdata['site_no'].unique()
    else:
        seg_id_nat = Stations
    if sites > len(seg_id_nat):
        sites = len(seg_id_nat)
    AA = random.sample(range(0, len(seg_id_nat)), sites)
    AA.sort()
    BB = [seg_id_nat[(x)] for x in AA]
    seg_id_nat = BB
    #seg_id_nat.sort()
    gage = [jj for jj in range(sites)]
    for i in range(1):
        gageindex = gage  #[i * 4:(i + 1) * 4]
        print(gageindex)
        t = utils.time.tRange2Array(tRange)
        fig, axes = plt.subplots(2, 1, figsize=(10.7, 13), constrained_layout=True)
        axes = axes.flat
        npred = len(predLst)  # 2  # plot the first two prediction: Base LSTM and DI(1)
        # subtitle = txt[i] + ' (Station ID:' + str(seg_id_nat[i]) + ') '
        # if i < (math.ceil(len(gage) / 4) - 1):
        #     subtitle = ['(a) (Station ID:' + str(seg_id_nat[4 * i]) + ') ',
        #                 '(b) (Station ID:' + str(seg_id_nat[4 * i + 1]) + ') ',
        #                 '(c) (Station ID:' + str(seg_id_nat[4 * i + 2]) + ') ',
        #                 '(d) (Station ID:' + str(seg_id_nat[4 * i + 3]) + ') ']
        # elif i == (math.ceil(len(gage) / 4) - 1):
        #     if ((len(gage)) - (i * 4)) == 1:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 2:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 3:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 2]) + ')']
        #txt = ['a', 'b', 'c', 'd']
        ylabel = 'Nitrate Concentration (mg/L)'

        for k in range(len(gageindex)):
            # iGrid = AA[gageindex[k]]
            iGrid = inputdata.index[inputdata['site_no'] == seg_id_nat[AA[gageindex[k]]]].values[0]
            yPlot = [obs[iGrid, :]]
            for y in predLst[0:npred]:
                yPlot.append(y[iGrid, :])
            # get the NSE value of LSTM and DI(1) model
            # Metrics = '[' + str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2)) +',\n'+str(np.round(statDictLst_res[0]['NSE'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr'][iGrid], 2)) + ',\n' +str(np.round(statDictLst_res[0]['NSE_res'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr_res'][iGrid], 2)) + ']'
            # subtitle1 = txt[k] + ' (Site: ' + str(seg_id_nat[k]) + ') ' + \
            subtitle1 =  ' (Site: ' + str(seg_id_nat[k]) + ') ' +\
                '[' + "RMSE: " + \
                      str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2))  +  \
                ', ' +  "NSE: " + str(np.round(statDictLst_res[0]['NSE'][iGrid], 2))  + \
                ', ' +  "Bias: " + str(np.round(statDictLst_res[0]['Bias'][iGrid], 2))  + \
                ', ' +  "Corr: " + str(np.round(statDictLst_res[0]['Corr'][iGrid], 2))  + \
                ', ' +  "KGE: " + str(np.round(statDictLst_res[0]['KGE'][iGrid], 2)) + \
                ']'
             #NSE_LSTM = [] #str(round(statDictLst[0]['NSE'][iGrid], 2))
            # NSE_DI1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
            # plot time series
            plotTS(
                t,
                yPlot,
                ax=axes[k],
                cLst='rkcbmkrmg',
                markerLst='o-----+1o-----+1',
                legLst=['Observed', "Hindcasted", 'Predicted_test', 'Predicted_train',
                        'Observed1',"Hindcasted1", 'Predicted_test1', 'Predicted_train1'],
                title=subtitle1, linespec=['o', '-', '-', '-', '-', ':', '+'],  #legLst=legLst=[TempTarget, 'LSTM: ' + NSE_LSTM], title=subtitle[k]
                ylabel=ylabel , figNo=k)  # ['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DI1]
        #boxPlotName = 'Time Series simulated and observed data in testing period- values in brackets are [RMSE, NSE, Bias, NSE_res, Corr_res]'
        fig.suptitle(boxPlotName, fontsize=12.2)

        #plotName = "TempSeries.eps"
        plotName = "TempSeries.png"


        if retrained is True:
            plt.savefig(os.path.join(rootOut, out_retrained, plotName), dpi=200)
            plt.savefig(os.path.join(rootOut, out_retrained, '-LowRes'+plotName))
        else:
            plt.savefig(os.path.join(rootOut, save_path, plotName), dpi=200, bbox_inches='tight' )
            plt.savefig(os.path.join(rootOut, save_path, '-LowRes'+plotName), bbox_inches='tight' )
        fig.show()
        
# def TempSeries_2_Plots_NO3_pred(attr_path, statDictLst_res, obs, predLst, TempTarget, tRange, boxPlotName, rootOut, save_path, sites=18, Stations=None, retrained=False):
def TempSeries_2_Plots_NO3_pred(attr_path, obs_np_train, obs_np_test,
                                pred_np_winter_total,
                                 pred_np_spring_total,
                                 pred_np_summer_total,
                                 pred_np_fall_total,
                                 TempTarget, tRange,
                                boxPlotName, rootOut, save_path,
                                sites=8, Stations=None, retrained=False):

    # fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    # axes = axes.flat
    npred = 2  # 2  # plot the first two prediction: Base LSTM and DI(1)
    #subtitle = ['(seg_id_nat:1450)', '(seg_id_nat:1566)', '(seg_id_nat:1718)', '(seg_id_nat:2013)']
    txt = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    ylabel = 'Nitrate Concentration (mg/L)'


    seg_id_nat = []
    inputdata = pd.read_feather(attr_path)
    # tRange = [20141001, 20161001]
    gage = []
    if Stations == None:
        seg_id_nat = inputdata['site_no'].unique()
    else:
        seg_id_nat = Stations
    if sites > len(seg_id_nat):
        sites = len(seg_id_nat)
    AA = random.sample(range(0, len(seg_id_nat)), sites)
    AA.sort()
    BB = [seg_id_nat[(x)] for x in AA]
    seg_id_nat = BB
    #seg_id_nat.sort()
    gage = [jj for jj in range(sites)]
    for i in range(1):
        gageindex = gage  #[i * 4:(i + 1) * 4]
        print(gageindex)
        t = utils.time.tRange2Array(tRange)
        fig, axes = plt.subplots(2, 1, figsize=(10.7, 13), constrained_layout=True)
        axes = axes.flat
        npred = 4   #len(predLst)  # 2  # plot the first two prediction: Base LSTM and DI(1)
        # subtitle = txt[i] + ' (Station ID:' + str(seg_id_nat[i]) + ') '
        # if i < (math.ceil(len(gage) / 4) - 1):
        #     subtitle = ['(a) (Station ID:' + str(seg_id_nat[4 * i]) + ') ',
        #                 '(b) (Station ID:' + str(seg_id_nat[4 * i + 1]) + ') ',
        #                 '(c) (Station ID:' + str(seg_id_nat[4 * i + 2]) + ') ',
        #                 '(d) (Station ID:' + str(seg_id_nat[4 * i + 3]) + ') ']
        # elif i == (math.ceil(len(gage) / 4) - 1):
        #     if ((len(gage)) - (i * 4)) == 1:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 2:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 3:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 2]) + ')']
        #txt = ['a', 'b', 'c', 'd']
        ylabel = 'Nitrate Concentration (mg/L)'

        for k in range(len(gageindex)):
            # iGrid = AA[gageindex[k]]
            iGrid = inputdata.index[inputdata['site_no'] == seg_id_nat[AA[gageindex[k]]]].values[0]
            yPlot = [obs_np_train[iGrid, :]]
            yPlot.append(obs_np_test[iGrid, :])

            yPlot.append(pred_np_winter_total[iGrid, :])
            yPlot.append(pred_np_spring_total[iGrid, :])
            yPlot.append(pred_np_summer_total[iGrid, :])
            yPlot.append(pred_np_fall_total[iGrid, :])

            # get the NSE value of LSTM and DI(1) model
            # Metrics = '[' + str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2)) +',\n'+str(np.round(statDictLst_res[0]['NSE'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr'][iGrid], 2)) + ',\n' +str(np.round(statDictLst_res[0]['NSE_res'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr_res'][iGrid], 2)) + ']'
            # subtitle1 = txt[k] + ' (Site: ' + str(seg_id_nat[k]) + ') ' + \
            subtitle1 = ' (Site: ' + str(seg_id_nat[k]) + ') '

             #NSE_LSTM = [] #str(round(statDictLst[0]['NSE'][iGrid], 2))
            # NSE_DI1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
            # plot time series
            plotTS(
                t,
                yPlot,
                ax=axes[k],
                #cLst='rkcbmkrmg',
                cLst=["red", "blue", "lightcoral", "turquoise", "violet", "lightgreen"],
                markerLst='o*-----+1o-----+1',
                legLst=['Obs training', "Obs test", 'Pred winter', 'Pred spring',
                        'Pred summer',"Pred fall"],
                title=subtitle1, linespec=['o', 'o', '-', '-', '-', '-', ':', '+'],  #legLst=legLst=[TempTarget, 'LSTM: ' + NSE_LSTM], title=subtitle[k]
                ylabel=ylabel , figNo=k)  # ['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DI1]
        #boxPlotName = 'Time Series simulated and observed data in testing period- values in brackets are [RMSE, NSE, Bias, NSE_res, Corr_res]'
        fig.suptitle(boxPlotName, fontsize=12.2)

        #plotName = "TempSeries.eps"
        plotName = "TempSeries.png"
        plotName1 = "TempSeries.eps"


        if retrained is True:
            plt.savefig(os.path.join(rootOut, out_retrained, plotName), dpi=200)
            plt.savefig(os.path.join(rootOut, out_retrained, '-LowRes'+plotName))
            plt.savefig(os.path.join(rootOut, out_retrained, plotName1), dpi=200)
            plt.savefig(os.path.join(rootOut, out_retrained, '-LowRes'+plotName1))
        else:
            plt.savefig(os.path.join(rootOut, save_path, plotName), dpi=200, bbox_inches='tight')
            plt.savefig(os.path.join(rootOut, save_path, '-LowRes'+plotName), bbox_inches='tight')
            plt.savefig(os.path.join(rootOut, save_path, plotName1), dpi=200, bbox_inches='tight')
            plt.savefig(os.path.join(rootOut, save_path, '-LowRes'+plotName1), bbox_inches='tight')
        fig.show()

def TempSeries_Seasonal_Plots_N03(attr_path, statDictLst_res, obs, obs_np_test, predLst, TempTarget, tRange, boxPlotName, rootOut, save_path, sites=18, Stations=None, retrained=False):  ## save_path,
    # fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    # axes = axes.flat
    npred = 2  # 2  # plot the first two prediction: Base LSTM and DI(1)
    #subtitle = ['(seg_id_nat:1450)', '(seg_id_nat:1566)', '(seg_id_nat:1718)', '(seg_id_nat:2013)']
    txt = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    ylabel = 'Nitrate Concentration (mg/L)'


    seg_id_nat = []
    inputdata = pd.read_feather(attr_path)
    # tRange = [20141001, 20161001]
    gage = []
    if Stations == None:
        seg_id_nat = inputdata['site_no'].unique()
    else:
        seg_id_nat = Stations
    if sites > len(seg_id_nat):
        sites = len(seg_id_nat)
    AA = random.sample(range(0, len(seg_id_nat)), sites)
    AA.sort()
    BB = [seg_id_nat[(x)] for x in AA]
    seg_id_nat = BB
    #seg_id_nat.sort()
    gage = [jj for jj in range(sites)]
    for i in range(1):
        gageindex = gage  #[i * 4:(i + 1) * 4]
        print(gageindex)
        t = utils.time.tRange2Array(tRange)
        

       
        fig, axes = plt.subplots(2, 1, figsize=(10.7, 13), constrained_layout=True)
        axes = axes.flat
        npred = len(predLst)  # 2  # plot the first two prediction: Base LSTM and DI(1)
        # subtitle = txt[i] + ' (Station ID:' + str(seg_id_nat[i]) + ') '
        # if i < (math.ceil(len(gage) / 4) - 1):
        #     subtitle = ['(a) (Station ID:' + str(seg_id_nat[4 * i]) + ') ',
        #                 '(b) (Station ID:' + str(seg_id_nat[4 * i + 1]) + ') ',
        #                 '(c) (Station ID:' + str(seg_id_nat[4 * i + 2]) + ') ',
        #                 '(d) (Station ID:' + str(seg_id_nat[4 * i + 3]) + ') ']
        # elif i == (math.ceil(len(gage) / 4) - 1):
        #     if ((len(gage)) - (i * 4)) == 1:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 2:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 3:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 2]) + ')']
        #txt = ['a', 'b', 'c', 'd']
        ylabel = 'Nitrate Concentration (mg/L)'

        for k in range(len(gageindex)):
            
            # iGrid = AA[gageindex[k]]
            iGrid = inputdata.index[inputdata['site_no'] == seg_id_nat[AA[gageindex[k]]]].values[0]
            yPlot = [obs[iGrid, :]]
            yPlot.append(obs_np_test[iGrid, :])
            for y in predLst[0:npred]:
                yPlot.append(y[iGrid, :])
            # get the NSE value of LSTM and DI(1) model
            # Metrics = '[' + str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2)) +',\n'+str(np.round(statDictLst_res[0]['NSE'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr'][iGrid], 2)) + ',\n' +str(np.round(statDictLst_res[0]['NSE_res'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr_res'][iGrid], 2)) + ']'
            # subtitle1 = txt[k] + ' (Site: ' + str(seg_id_nat[k]) + ') ' + \
            subtitle1 =  ' (Site: ' + str(seg_id_nat[k]) + ') ' +\
                '[' + "RMSE: " + \
                      str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2))  +  \
                ', ' +  "NSE: " + str(np.round(statDictLst_res[0]['NSE'][iGrid], 2))  + \
                ', ' +  "Bias: " + str(np.round(statDictLst_res[0]['Bias'][iGrid], 2))  + \
                ', ' +  "Corr: " + str(np.round(statDictLst_res[0]['Corr'][iGrid], 2))  + \
                ', ' +  "KGE: " + str(np.round(statDictLst_res[0]['KGE'][iGrid], 2)) + \
                ']'
             #NSE_LSTM = [] #str(round(statDictLst[0]['NSE'][iGrid], 2))
            # NSE_DI1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
            # plot time series
            plotTS_NO3(
                t,
                yPlot,
                ax=axes[k],
                # cLst='rkcbmkrmg',
                cLst = ["red", "blue", "lightgreen", "lightcoral", "turquoise", "violet"],
                markerLst='o*-----+1o-----+1',
                legLst=['Obs_train', "obs_test", 'winter', 'spring',
                        'summer',"fall"],
                title=subtitle1, linespec=['o', '*', '-', '-', '-', '-', ':', '+'],  #legLst=legLst=[TempTarget, 'LSTM: ' + NSE_LSTM], title=subtitle[k]
                ylabel=ylabel , figNo=k)  # ['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DI1]
        #boxPlotName = 'Time Series simulated and observed data in testing period- values in brackets are [RMSE, NSE, Bias, NSE_res, Corr_res]'
        fig.suptitle(boxPlotName, fontsize=12.2)

        plotName_eps = "TempSeries.eps"
        plotName_png = "TempSeries.png"


        if retrained is True:
            plt.savefig(os.path.join(rootOut, save_path, out_retrained, plotName_png), dpi=200) ## save_path,
            plt.savefig(os.path.join(rootOut, save_path, out_retrained, '-LowRes'+plotName_png)) ## save_path,
            plt.savefig(os.path.join(rootOut, save_path, out_retrained, plotName_eps), dpi=200) ## save_path,
            plt.savefig(os.path.join(rootOut, save_path, out_retrained, '-LowRes'+plotName_eps)) ## save_path,
        else:
            plt.savefig(os.path.join(rootOut, save_path, plotName_png), dpi=200, bbox_inches='tight')  ## save_path,
            plt.savefig(os.path.join(rootOut, save_path, '-LowRes'+plotName_png), bbox_inches='tight') ## save_path,
            plt.savefig(os.path.join(rootOut, save_path, plotName_eps), dpi=200, bbox_inches='tight') ## save_path,
            plt.savefig(os.path.join(rootOut, save_path, '-LowRes'+plotName_eps), bbox_inches='tight') ## save_path,
        fig.show()


def plotMultiBoxFig(data,
               label1=None,
               label2=None,
               colorLst='grbkcmy',
               title=None,
               figsize=(10, 8),
               sharey=True,
               xticklabel=None,
               position=None,
               ylabel=None
               ):
    nc = len(data)
    fig, axes = plt.subplots(ncols=nc, sharey=sharey, figsize=figsize, constrained_layout=True)
    nv = len(data[0])
    #ndays = len(data[0][1])
    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        bp = [None]*nv
        for ii in range(nv):
            bp[ii] = ax.boxplot(
            data[k][ii], patch_artist=True, notch=True, showfliers=False, positions=position[ii], widths=0.2)
            for kk in range(0, len(bp[ii]['boxes'])):
                plt.setp(bp[ii]['boxes'][kk], facecolor=colorLst[ii])

        if label1 is not None:
            ax.set_xlabel(label1[k])
        else:
            ax.set_xlabel(str(k))
        if ylabel is not None:
            ax.set_ylabel(ylabel[k])
        if xticklabel is None:
            ax.set_xticks([])
        else:
            ax.set_xticks([y for y in range(0,len(data[k][1])+1)])
            ax.set_xticklabels(xticklabel)
        ax.set_xlim([-0.5, ndays+0.5])
        # ax.ticklabel_format(axis='y', style='sci')
        vlabel = np.arange(0.5, len(data[k][1])+1)
        for xv in vlabel:
            ax.axvline(xv, ymin=0, ymax=1, color='k',
                       linestyle='dashed', linewidth=1)
        yh = np.nanmedian(data[k][0][0])
        ax.axhline(yh, xmin=0, xmax=1, color='r',
                   linestyle='dashed', linewidth=2)
        yh1 = np.nanmedian(data[k][1][0])
        ax.axhline(yh1, xmin=0, xmax=1, color='b',
                   linestyle='dashed', linewidth=2)
    labelhandle = list()
    for ii in range(nv):
        labelhandle.append(bp[ii]['boxes'][0])
    if label2 is not None:
        if nc == 1:
            ax.legend(labelhandle, label2, loc='best', frameon=False, ncol=1)
        else:
            axes[-1].legend(labelhandle, label2, loc='best', frameon=False, ncol=1, fontsize=12)
    if title is not None:
        fig.suptitle(title)
    return fig


def plotTS(t,
           y,
           *,
           ax=None,
           tBar=None,
           figsize=(12, 4),
           cLst='rbkgcmy',
           markerLst=None,
           linespec=None,
           legLst=None,
           title=None,
           linewidth=2,
           ylabel=None,
           figNo=None):
    newFig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
        newFig = True

    if type(y) is np.ndarray:
        y = [y]
    for k in range(len(y)):
        tt = t[k] if type(t) is list else t
        yy = y[k]
        legStr = None
        if legLst is not None:
            legStr = legLst[k]
        if markerLst is None:
            if True in np.isnan(yy):
                ax.plot(tt, yy, '*', color=cLst[k], label=legStr)
            else:
                ax.plot(
                    tt, yy, color=cLst[k], label=legStr, linewidth=linewidth)
        else:
            if markerLst[k] is '-':
                if linespec is not None:
                    ax.plot(tt, yy, color=cLst[k], label=legStr, linestyle=linespec[k], lw=1.15)
                else:
                    ax.plot(tt, yy, color=cLst[k], label=legStr, lw=1.15)
            else:
                ax.scatter(
                    tt, yy, color=cLst[k], label=legStr, marker=markerLst[k], s=5)
        if ylabel is not None:
            if figNo % 2 == 0:
                ax.set_ylabel(ylabel)
        #ax.set_xlim([np.min(tt), np.max(tt)])
    if tBar is not None:
        ylim = ax.get_ylim()
        tBar = [tBar] if type(tBar) is not list else tBar
        for tt in tBar:
            ax.plot([tt, tt], ylim, '-k')

    if legLst is not None:
        ax.legend(loc='lower right', frameon=False)
    if title is not None:
        ax.set_title(title, loc='center', fontsize=10.5)
    #ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    #ax.set_xticks([np.min(tt), np.min(tt) + 180, np.min(tt)+365, np.max(tt)], [np.min(tt), np.min(tt) + 180, np.min(tt)+365, np.max(tt)])
    #ax.set_xticklabels( [np.min(tt), np.min(tt) + 90, np.min(tt) + 180, np.min(tt) + 270 , np.min(tt)+365, np.max(tt)], fontsize=8)
    for tick in ax.xaxis.get_major_ticks():

        tick.label.set_fontsize(11)
        # specify integer or one of preset strings, e.g.
        # tick.label.set_fontsize('x-small')
        #tick.label.set_rotation('vertical')
    if newFig is True:
        return fig, ax
    else:
        return ax

def plotTS_NO3(t,
           y,
           *,
           ax=None,
           tBar=None,
           figsize=(12, 4),
           cLst='rbkgcmy',
           markerLst=None,
           linespec=None,
           legLst=None,
           title=None,
           linewidth=2,
           ylabel=None,
           figNo=None):
    newFig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
        newFig = True

    if type(y) is np.ndarray:
        y = [y]
    for k in range(len(y)):
        tt = t[k] if type(t) is list else t
        yy = y[k]
        legStr = None
        if legLst is not None:
            legStr = legLst[k]
        if markerLst is None:
            if True in np.isnan(yy):
                ax.plot(tt, yy, '*', color=cLst[k], label=legStr)
            else:
                ax.plot(
                    tt, yy, color=cLst[k], label=legStr, linewidth=linewidth)
        else:
            if markerLst[k] is '-':
                if linespec is not None:
                    ax.plot(tt, yy, color=cLst[k], label=legStr, linestyle=linespec[k], lw=1.15)
                else:
                    ax.plot(tt, yy, color=cLst[k], label=legStr, lw=1.15)
            else:
                ax.scatter(
                    tt, yy, color=cLst[k], label=legStr, marker=markerLst[k], s=5)
        if ylabel is not None:
            if figNo % 2 == 0:
                ax.set_ylabel(ylabel)
        #ax.set_xlim([np.min(tt), np.max(tt)])
    if tBar is not None:
        ylim = ax.get_ylim()
        tBar = [tBar] if type(tBar) is not list else tBar
        for tt in tBar:
            ax.plot([tt, tt], ylim, '-k')

    if legLst is not None:
        ax.legend(loc='lower right', frameon=False)
    if title is not None:
        ax.set_title(title, loc='center', fontsize=10.5)
    #ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    #ax.set_xticks([np.min(tt), np.min(tt) + 180, np.min(tt)+365, np.max(tt)], [np.min(tt), np.min(tt) + 180, np.min(tt)+365, np.max(tt)])
    #ax.set_xticklabels( [np.min(tt), np.min(tt) + 90, np.min(tt) + 180, np.min(tt) + 270 , np.min(tt)+365, np.max(tt)], fontsize=8)
    for tick in ax.xaxis.get_major_ticks():

        tick.label.set_fontsize(11)
        # specify integer or one of preset strings, e.g.
        # tick.label.set_fontsize('x-small')
        #tick.label.set_rotation('vertical')
    if newFig is True:
        return fig, ax
    else:
        return ax


def plotVS(x,
           y,
           *,
           ax=None,
           title=None,
           xlabel=None,
           ylabel=None,
           titleCorr=True,
           plot121=True,
           doRank=False,
           figsize=(8, 6)):
    if doRank is True:
        x = scipy.stats.rankdata(x)
        y = scipy.stats.rankdata(y)
    corr = scipy.stats.pearsonr(x, y)[0]
    pLr = np.polyfit(x, y, 1)
    xLr = np.array([np.min(x), np.max(x)])
    yLr = np.poly1d(pLr)(xLr)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = None
    if title is not None:
        if titleCorr is True:
            title = title + ' ' + r'$\rho$={:.2f}'.format(corr)
        ax.set_title(title)
    else:
        if titleCorr is True:
            ax.set_title(r'$\rho$=' + '{:.2f}'.format(corr))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # corr = np.corrcoef(x, y)[0, 1]
    ax.plot(x, y, 'b.')
    ax.plot(xLr, yLr, 'r-')

    if plot121 is True:
        plot121Line(ax)

    return fig, ax


def plot121Line(ax, spec='k-'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vmin = np.min([xlim[0], ylim[0]])
    vmax = np.max([xlim[1], ylim[1]])
    ax.plot([vmin, vmax], [vmin, vmax], spec)


def plotMap(data,
            *,
            ax=None,
            lat=None,
            lon=None,
            title=None,
            cRange=None,
            shape=None,
            pts=None,
            figsize=(22, 11),
            clbar=True,
            cRangeint=False,
            cmap=plt.cm.jet,
            bounding=None,
            prj='cyl'):

    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        temp = flatData(data)
        vmin = np.percentile(temp, 5)
        vmax = np.percentile(temp, 95)
        if cRangeint is True:
            vmin = int(round(vmin))
            vmax = int(round(vmax))
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    if len(data.squeeze().shape) == 1:
        isGrid = False
    else:
        isGrid = True
    if bounding is None:
        bounding = [np.min(lat)-1.5, np.max(lat)+0.5,
                    np.min(lon)-1.5, np.max(lon)+0.5]

    mm = basemap.Basemap(
        llcrnrlat=bounding[0],
        urcrnrlat=bounding[1],
        llcrnrlon=bounding[2],
        urcrnrlon=bounding[3],
        projection=prj,
        resolution='c',
        ax=ax)
    mm.drawcoastlines()
    mm.drawstates(linestyle='dashed')
    mm.drawcountries(linewidth=1.0, linestyle='-.')
    x, y = mm(lon, lat)
    if isGrid is True:
        xx, yy = np.meshgrid(x, y)
        cs = mm.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)
        # cs = mm.imshow(
        #     np.flipud(data),
        #     cmap=plt.cm.jet(np.arange(0, 1, 0.1)),
        #     vmin=vmin,
        #     vmax=vmax,
        #     extent=[x[0], x[-1], y[0], y[-1]])
    else:
        cs = mm.scatter(
            x, y, c=data, s=120, cmap=plt.cm.jet_r, vmin=vmin, vmax=vmax)

    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                mm.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            mm.plot(x, y, color='r', linewidth=3)
    if pts is not None:
        mm.plot(pts[1], pts[0], 'k*', markersize=4)
        npt = len(pts[0])
        for k in range(npt):
            plt.text(
                pts[1][k],
                pts[0][k],
                string.ascii_uppercase[k],
                fontsize=18)
    if clbar is True:
        cbar = mm.colorbar(cs, pad='1%')
        cbar.ax.tick_params(labelsize=20)
    if title is not None:
        ax.set_title(title, fontsize=26)

    ax.legend(handles=[cs],
            labels=['$\ LSTM_{flow}$'],
            loc='lower right',
            fontsize=22)
    if ax is None:
        return fig, ax, mm
    else:
        return mm, cs


def plotlocmap(
            lat,
            lon,
            ax=None,
            baclat=None,
            baclon=None,
            title=None,
            shape=None,
            txtlabel=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots()
    mm = basemap.Basemap(
        llcrnrlat=min(np.min(baclat),np.min(lat))-0.5,
        urcrnrlat=max(np.max(baclat),np.max(lat))+0.5,
        llcrnrlon=min(np.min(baclon),np.min(lon))-0.5,
        urcrnrlon=max(np.max(baclon),np.max(lon))+0.5,
        projection='cyl',
        resolution='c',
        ax=ax)
    mm.drawcoastlines()
    mm.drawstates(linestyle='dashed')
    mm.drawcountries(linewidth=1.0, linestyle='-.')
    # x, y = mm(baclon, baclat)
    # bs = mm.scatter(
    #     x, y, c='k', s=30)
    x, y = mm(lon, lat)
    ax.plot(x, y, 'k*', markersize=12)
    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                mm.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            mm.plot(x, y, color='r', linewidth=3)
    if title is not None:
        ax.set_title(title, loc='left')
    if txtlabel is not None:
        for ii in range(len(lat)):
            txt = txtlabel[ii]
            xy = (x[ii], y[ii])
            xy = (x[ii]+1.0, y[ii]-1.5)
            ax.annotate(txt, xy, fontsize=16, fontweight='bold')
        if ax is None:
            return fig, ax, mm
        else:
            return mm


def plotPUBloc(data,
            *,
            ax=None,
            lat=None,
            lon=None,
            baclat=None,
            baclon=None,
            title=None,
            cRange=None,
            cRangeint=False,
            shape=None):
    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        temp = flatData(data)
        vmin = np.percentile(temp, 5)
        vmax = np.percentile(temp, 95)
        if cRangeint is True:
            vmin = int(round(vmin))
            vmax = int(round(vmax))
    if ax is None:
        # fig, ax = plt.figure(figsize=(8, 4))
        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots()
    if len(data.squeeze().shape) == 1:
        isGrid = False
    else:
        isGrid = True

    mm = basemap.Basemap(
        llcrnrlat=min(np.min(baclat),np.min(lat))-0.5,
        urcrnrlat=max(np.max(baclat),np.max(lat))+0.5,
        llcrnrlon=min(np.min(baclon),np.min(lon))-0.5,
        urcrnrlon=max(np.max(baclon),np.max(lon))+0.5,
        projection='cyl',
        resolution='c',
        ax=ax)
    mm.drawcoastlines()
    mm.drawstates(linestyle='dashed')
    mm.drawcountries(linewidth=0.5, linestyle='-.')
    x, y = mm(baclon, baclat)
    bs = mm.scatter(
        x, y, c='k', s=30)
    x, y = mm(lon, lat)
    if isGrid is True:
        xx, yy = np.meshgrid(x, y)
        cs = mm.pcolormesh(xx, yy, data, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    else:
        cs = mm.scatter(
            x, y, c=data, s=100, cmap=plt.cm.jet, vmin=vmin, vmax=vmax, marker='*')

    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                mm.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            mm.plot(x, y, color='r', linewidth=3)
    mm.colorbar(cs, location='bottom', pad='5%')
    if title is not None:
        ax.set_title(title)
        if ax is None:
            return fig, ax, mm
        else:
            return mm

def plotTsMap(dataMap,
              dataTs,
              *,
              lat,
              lon,
              t,
              dataTs2=None,
              tBar=None,
              mapColor=None,
              tsColor='krbg',
              tsColor2='cmy',
              mapNameLst=None,
              tsNameLst=None,
              tsNameLst2=None,
              figsize=[12, 6],
              isGrid=False,
              multiTS=False,
              linewidth=1):
    if type(dataMap) is np.ndarray:
        dataMap = [dataMap]
    if type(dataTs) is np.ndarray:
        dataTs = [dataTs]
    if dataTs2 is not None:
        if type(dataTs2) is np.ndarray:
            dataTs2 = [dataTs2]
    nMap = len(dataMap)

    # setup axes
    fig = plt.figure(figsize=figsize)
    if multiTS is False:
        nAx = 1
        dataTs = [dataTs]
        if dataTs2 is not None:
            dataTs2 = [dataTs2]
    else:
        nAx = len(dataTs)
    gs = gridspec.GridSpec(3 + nAx, nMap)
    gs.update(wspace=0.025, hspace=0)
    axTsLst = list()
    for k in range(nAx):
        axTs = fig.add_subplot(gs[k + 3, :])
        axTsLst.append(axTs)
    if dataTs2 is not None:
        axTs2Lst = list()
        for axTs in axTsLst:
            axTs2 = axTs.twinx()
            axTs2Lst.append(axTs2)

    # plot maps
    for k in range(nMap):
        ax = fig.add_subplot(gs[0:2, k])
        cRange = None if mapColor is None else mapColor[k]
        title = None if mapNameLst is None else mapNameLst[k]
        data = dataMap[k]
        if isGrid is False:
            plotMap(data, lat=lat, lon=lon, ax=ax, cRange=cRange, title=title)
        else:
            grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
            plotMap(grid, lat=uy, lon=ux, ax=ax, cRange=cRange, title=title)

    # plot ts
    def onclick(event):
        xClick = event.xdata
        yClick = event.ydata
        d = np.sqrt((xClick - lon)**2 + (yClick - lat)**2)
        ind = np.argmin(d)
        # titleStr = 'pixel %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
#         titleStr = 'gage %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
#         ax.clear()
#         plotMap(data, lat=lat, lon=lon, ax=ax, cRange=cRange, title=title)
#         ax.plot(lon[ind], lat[ind], 'k*', markersize=12)
        titleStr = 'pixel %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
        for ix in range(nAx):
            tsLst = list()
            for temp in dataTs[ix]:
                tsLst.append(temp[ind, :])
            axTsLst[ix].clear()
            if ix == 0:
                plotTS(
                    t,
                    tsLst,
                    ax=axTsLst[ix],
                    legLst=tsNameLst,
                    title=titleStr,
                    cLst=tsColor,
                    linewidth=linewidth,
                    tBar=tBar)
            else:
                plotTS(
                    t,
                    tsLst,
                    ax=axTsLst[ix],
                    legLst=tsNameLst,
                    cLst=tsColor,
                    linewidth=linewidth,
                    tBar=tBar)

            if dataTs2 is not None:
                tsLst2 = list()
                for temp in dataTs2[ix]:
                    tsLst2.append(temp[ind, :])
                axTs2Lst[ix].clear()
                plotTS(
                    t,
                    tsLst2,
                    ax=axTs2Lst[ix],
                    legLst=tsNameLst2,
                    cLst=tsColor2,
                    lineWidth=linewidth,
                    tBar=tBar)
            if ix != nAx - 1:
                axTsLst[ix].set_xticklabels([])
        plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

def plotTsMapGage(dataMap,
              dataTs,
              *,
              lat,
              lon,
              t,
              colorMap=None,
              mapNameLst=None,
              tsNameLst=None,
              figsize=[12, 6]):
    if type(dataMap) is np.ndarray:
        dataMap = [dataMap]
    if type(dataTs) is np.ndarray:
        dataTs = [dataTs]
    nMap = len(dataMap)
    nTs = len(dataTs)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(3, nMap)

    for k in range(nMap):
        ax = fig.add_subplot(gs[0:2, k])
        cRange = None if colorMap is None else colorMap[k]
        title = None if mapNameLst is None else mapNameLst[k]
        data = dataMap[k]
        if len(data.squeeze().shape) == 1:
            plotMap(data, lat=lat, lon=lon, ax=ax, cRange=cRange, title=title)
        else:
            grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
            plotMap(grid, lat=uy, lon=ux, ax=ax, cRange=cRange, title=title)
    axTs = fig.add_subplot(gs[2, :])

    def onclick(event):
        xClick = event.xdata
        yClick = event.ydata
        d = np.sqrt((xClick - lon)**2 + (yClick - lat)**2)
        ind = np.argmin(d)
        # titleStr = 'pixel %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
        titleStr = 'gage %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
        ax.clear()
        plotMap(data, lat=lat, lon=lon, ax=ax, cRange=cRange, title=title)
        ax.plot(lon[ind], lat[ind], 'k*', markersize=12)
        # ax.draw(renderer=None)
        tsLst = list()
        for k in range(nTs):
            tsLst.append(dataTs[k][ind, :])
        axTs.clear()
        plotTS(t, tsLst, ax=axTs, legLst=tsNameLst, title=titleStr)
        plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()


def plotCDF(xLst,
            *,
            ax=None,
            title=None,
            legendLst=None,
            figsize=(8, 6),
            ref='121',
            cLst=None,
            xlabel=None,
            ylabel=None,
            showDiff='RMSE',
            xlim=None,
            linespec=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = None

    if cLst is None:
        cmap = plt.cm.jet
        cLst = cmap(np.linspace(0, 1, len(xLst)))

    if title is not None:
        ax.set_title(title, loc='left')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    xSortLst = list()
    rmseLst = list()
    ksdLst = list()
    for k in range(0, len(xLst)):
        x = xLst[k]
        xSort = flatData(x)
        yRank = np.arange(len(xSort)) / float(len(xSort) - 1)
        xSortLst.append(xSort)
        if legendLst is None:
            legStr = None
        else:
            legStr = legendLst[k]
        if ref is not None:
            if ref is '121':
                yRef = yRank
            elif ref is 'norm':
                yRef = scipy.stats.norm.cdf(xSort, 0, 1)
            rmse = np.sqrt(((xSort - yRef)**2).mean())
            ksd = np.max(np.abs(xSort - yRef))
            rmseLst.append(rmse)
            ksdLst.append(ksd)
            if showDiff is 'RMSE':
                legStr = legStr + ' RMSE=' + '%.3f' % rmse
            elif showDiff is 'KS':
                legStr = legStr + ' KS=' + '%.3f' % ksd
        ax.plot(xSort, yRank, color=cLst[k], label=legStr, linestyle=linespec[k])
        ax.grid(b=True)
    if xlim is not None:
        ax.set(xlim=xlim)
    if ref is '121':
        ax.plot([0, 1], [0, 1], 'k', label='y=x')
    if ref is 'norm':
        xNorm = np.linspace(-5, 5, 1000)
        normCdf = scipy.stats.norm.cdf(xNorm, 0, 1)
        ax.plot(xNorm, normCdf, 'k', label='Gaussian')
    if legendLst is not None:
        ax.legend(loc='best', frameon=False)
    # out = {'xSortLst': xSortLst, 'rmseLst': rmseLst, 'ksdLst': ksdLst}
    return fig, ax

def flatData(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return (xSort)


def scaleSigma(s, u, y):
    yNorm = (y - u) / s
    _, sF = scipy.stats.norm.fit(flatData(yNorm))
    return sF


def reCalSigma(s, u, y):
    conf = scipy.special.erf(np.abs(y - u) / s / np.sqrt(2))
    yNorm = (y - u) / s
    return conf, yNorm


def regLinear(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    out = sm.OLS(y, X).fit()
    return out


def TempSeries_4_Plots(attr_path, statDictLst_res, obs, predLst, TempTarget, tRange, boxPlotName, rootOut, save_path, sites=18, Stations=None, retrained=False):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.flat
    npred = 2  # 2  # plot the first two prediction: Base LSTM and DI(1)
    #subtitle = ['(seg_id_nat:1450)', '(seg_id_nat:1566)', '(seg_id_nat:1718)', '(seg_id_nat:2013)']
    txt = ['a', 'b', 'c', 'd']
    ylabel = 'Stream Temperature ($\mathregular{deg}$ C)'


    seg_id_nat = []
    inputdata = pd.read_feather(attr_path)
    # tRange = [20141001, 20161001]
    gage = []
    if Stations == None:
        seg_id_nat = inputdata['site_no'].unique()
    else:
        seg_id_nat = Stations
    if sites > len(seg_id_nat):
        sites = len(seg_id_nat)
    AA = random.sample(range(0, len(seg_id_nat)), sites)
    AA.sort()
    BB = [seg_id_nat[(x)] for x in AA]
    seg_id_nat = BB
    #seg_id_nat.sort()
    gage = [jj for jj in range(sites)]
    for i in range(math.ceil(len(gage) / 4)):
        gageindex = gage[i * 4:(i + 1) * 4]
        print(gageindex)
        t = utils.time.tRange2Array(tRange)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        axes = axes.flat
        npred = 2  # 2  # plot the first two prediction: Base LSTM and DI(1)
        if i < (math.ceil(len(gage) / 4) - 1):
            subtitle = ['(a) (Station ID:' + str(seg_id_nat[4 * i]) + ') ',
                        '(b) (Station ID:' + str(seg_id_nat[4 * i + 1]) + ') ',
                        '(c) (Station ID:' + str(seg_id_nat[4 * i + 2]) + ') ',
                        '(d) (Station ID:' + str(seg_id_nat[4 * i + 3]) + ') ']
        elif i == (math.ceil(len(gage) / 4) - 1):
            if ((len(gage)) - (i * 4)) == 1:
                subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')']
            elif ((len(gage)) - (i * 4)) == 2:
                subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
                    , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')']
            elif ((len(gage)) - (i * 4)) == 3:
                subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
                    , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')'
                    , '(Station ID:' + str(seg_id_nat[4 * i + 2]) + ')']
        txt = ['a', 'b', 'c', 'd']
        ylabel = 'Stream Temperature ($\mathregular{deg}$ C)'

        for k in range(len(gageindex)):
            iGrid = AA[gageindex[k]]
            iGrid = inputdata.index[inputdata['site_no'] == seg_id_nat[AA[gageindex[k]]]].values[0]
            yPlot = [obs[iGrid, :]]
            for y in predLst[0:npred]:
                yPlot.append(y[iGrid, :])
            # get the NSE value of LSTM and DI(1) model
            Metrics = '[' + str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2)) +',\n'+str(np.round(statDictLst_res[0]['NSE'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr'][iGrid], 2)) + ',\n' +str(np.round(statDictLst_res[0]['NSE_res'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr_res'][iGrid], 2)) + ']'
            subtitle1 = '[' + str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2)) + ',' + str(
                np.round(statDictLst_res[0]['NSE'][iGrid], 2)) + ',' + str(
                np.round(statDictLst_res[0]['Corr'][iGrid], 2)) + ',' + str(
                np.round(statDictLst_res[0]['NSE_res'][iGrid], 2)) + ','+ str(
                np.round(statDictLst_res[0]['Corr_res'][iGrid], 2)) + ']'
             #NSE_LSTM = [] #str(round(statDictLst[0]['NSE'][iGrid], 2))
            # NSE_DI1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
            # plot time series
            plotTS(
                t,
                yPlot,
                ax=axes[k],
                cLst='bkrmg',
                markerLst='o-',
                legLst=['obs', 'Sim'], title=subtitle[k]+subtitle1, linespec=['o', '-', ':'],  #legLst=[TempTarget, 'LSTM: ' + NSE_LSTM], title=subtitle[k]
                ylabel=ylabel)  # ['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DI1]
        boxPlotName = 'Time Series simulated and observed data in testing period- values in brackets are [RMSE, NSE, Corr, NSE_res, Corr_res]'
        fig.suptitle(boxPlotName, fontsize=15)

        plotName = 'Fig' + str(i) + '(' + TempTarget + ')' + "Temp.png"

        # if retrained == True:
        #    # plt.savefig(os.path.join(rootOut, out_retrained, plotName), dpi=500)
        #     plt.savefig(os.path.join(rootOut, out_retrained, '-LowRes'+plotName))
        # else:
        #  #   plt.savefig(os.path.join(rootOut, save_path, plotName), dpi=500)
        #     plt.savefig(os.path.join(rootOut, save_path, '-LowRes'+plotName))
        fig.show()


def TempSeries_4_Plots_ERL(attr_path, statDictLst_res, obs, predLst, TempTarget, tRange, boxPlotName, rootOut, save_path, sites=18, Stations=None, retrained=False):
    # fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    # axes = axes.flat
    npred = 2  # 2  # plot the first two prediction: Base LSTM and DI(1)
    #subtitle = ['(seg_id_nat:1450)', '(seg_id_nat:1566)', '(seg_id_nat:1718)', '(seg_id_nat:2013)']
    txt = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    ylabel = 'Nitrate Concentration (mg/L)'


    seg_id_nat = []
    inputdata = pd.read_feather(attr_path)
    # tRange = [20141001, 20161001]
    gage = []
    if Stations == None:
        seg_id_nat = inputdata['site_no'].unique()
    else:
        seg_id_nat = Stations
    if sites > len(seg_id_nat):
        sites = len(seg_id_nat)
    AA = random.sample(range(0, len(seg_id_nat)), sites)
    AA.sort()
    BB = [seg_id_nat[(x)] for x in AA]
    seg_id_nat = BB
    #seg_id_nat.sort()
    gage = [jj for jj in range(sites)]
    for i in range(1):
        gageindex = gage  #[i * 4:(i + 1) * 4]
        print(gageindex)
        t = utils.time.tRange2Array(tRange)
        fig, axes = plt.subplots(4, 2, figsize=(10.7, 13), constrained_layout=True)
        axes = axes.flat
        npred = 1  # 2  # plot the first two prediction: Base LSTM and DI(1)
        # subtitle = txt[i] + ' (Station ID:' + str(seg_id_nat[i]) + ') '
        # if i < (math.ceil(len(gage) / 4) - 1):
        #     subtitle = ['(a) (Station ID:' + str(seg_id_nat[4 * i]) + ') ',
        #                 '(b) (Station ID:' + str(seg_id_nat[4 * i + 1]) + ') ',
        #                 '(c) (Station ID:' + str(seg_id_nat[4 * i + 2]) + ') ',
        #                 '(d) (Station ID:' + str(seg_id_nat[4 * i + 3]) + ') ']
        # elif i == (math.ceil(len(gage) / 4) - 1):
        #     if ((len(gage)) - (i * 4)) == 1:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 2:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')']
        #     elif ((len(gage)) - (i * 4)) == 3:
        #         subtitle = ['(Station ID:' + str(seg_id_nat[4 * i + 0]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 1]) + ')'
        #             , '(Station ID:' + str(seg_id_nat[4 * i + 2]) + ')']
        #txt = ['a', 'b', 'c', 'd']
        ylabel = 'Nitrate Concentration (mg/L)'

        for k in range(len(gageindex)):
            # iGrid = AA[gageindex[k]]
            iGrid = inputdata.index[inputdata['site_no'] == seg_id_nat[AA[gageindex[k]]]].values[0]
            yPlot = [obs[iGrid, :]]
            for y in predLst[0:npred]:
                yPlot.append(y[iGrid, :])
            # get the NSE value of LSTM and DI(1) model
            # Metrics = '[' + str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2)) +',\n'+str(np.round(statDictLst_res[0]['NSE'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr'][iGrid], 2)) + ',\n' +str(np.round(statDictLst_res[0]['NSE_res'][iGrid], 2)) + ',' +str(np.round(statDictLst_res[0]['Corr_res'][iGrid], 2)) + ']'
            subtitle1 = txt[k] + ' (USGS streamgage: ' + str(seg_id_nat[k]) + ') ' +\
                '[' + \
                      str(np.round(statDictLst_res[0]['RMSE'][iGrid], 2)) + \
                ',' + str(np.round(statDictLst_res[0]['NSE'][iGrid], 2)) + \
                ',' + str(np.round(statDictLst_res[0]['Bias'][iGrid], 2)) + \
                ',' + str(np.round(statDictLst_res[0]['Corr'][iGrid], 2)) + \
                ',' + str(np.round(statDictLst_res[0]['R2'][iGrid], 2)) + \
                ']'
             #NSE_LSTM = [] #str(round(statDictLst[0]['NSE'][iGrid], 2))
            # NSE_DI1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
            # plot time series
            plotTS(
                t,
                yPlot,
                ax=axes[k],
                cLst='bkrmg',
                markerLst='o---+1',
                legLst=['obs', 'predict'], title=subtitle1, linespec=['o', '-', '-', '-', ':', '+'],  #legLst=[TempTarget, 'LSTM: ' + NSE_LSTM], title=subtitle[k]
                ylabel=ylabel , figNo=k)  # ['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DI1]
        #boxPlotName = 'Time Series simulated and observed data in testing period- values in brackets are [RMSE, NSE, Bias, NSE_res, Corr_res]'
        fig.suptitle(boxPlotName, fontsize=12.2)

        plotName = "TempSeries.png"

        if retrained is True:
            plt.savefig(os.path.join(rootOut, out_retrained, plotName), dpi=200)
            plt.savefig(os.path.join(rootOut, out_retrained, '-LowRes'+plotName))
        else:
            plt.savefig(os.path.join(rootOut, save_path, plotName), dpi=200, bbox_inches='tight' )
            plt.savefig(os.path.join(rootOut, save_path, '-LowRes'+plotName), bbox_inches='tight' )
        fig.show()



def plotMap_separate(data, ind99_dam, ind99_nodam,
                      ind60_99_dam, ind60_99_nodam,
                      ind10_60_dam, ind10_60_nodam,

            *,
            ax=None,
            lat=None,
            lon=None,
            title=None,
            cRange=None,
            shape=None,
            pts=None,
            figsize=(22, 11),
            clbar=True,
            cRangeint=False,
            cmap=plt.cm.jet,
            bounding=None,
            prj='cyl',
            CMAP=plt.cm.jet,
            dataPUB=None,
            lat_PUB=None,
            lon_PUB=None):

    data99_dam = []
    lon99_dam = []
    lat99_dam = []
    for i, j in enumerate(ind99_dam):
        data99_dam.append(data[j])
        lon99_dam.append(lon[j])
        lat99_dam.append(lat[j])
    data60_99_dam = []
    lon60_99_dam = []
    lat60_99_dam = []
    for i,j in enumerate(ind60_99_dam):
        data60_99_dam.append(data[j])
        lon60_99_dam.append(lon[j])
        lat60_99_dam.append(lat[j])
    data10_60_dam = []
    lon10_60_dam = []
    lat10_60_dam = []
    for i, j in enumerate(ind10_60_dam):
        data10_60_dam.append(data[j])
        lon10_60_dam.append(lon[j])
        lat10_60_dam.append(lat[j])
    #############
    data99_nodam = []
    lon99_nodam = []
    lat99_nodam = []
    for i, j in enumerate(ind99_nodam):
        data99_nodam.append(data[j])
        lon99_nodam.append(lon[j])
        lat99_nodam.append(lat[j])
    data60_99_nodam = []
    lon60_99_nodam = []
    lat60_99_nodam = []
    for i, j in enumerate(ind60_99_nodam):
        data60_99_nodam.append(data[j])
        lon60_99_nodam.append(lon[j])
        lat60_99_nodam.append(lat[j])
    data10_60_nodam = []
    lon10_60_nodam = []
    lat10_60_nodam = []
    for i, j in enumerate(ind10_60_nodam):
        data10_60_nodam.append(data[j])
        lon10_60_nodam.append(lon[j])
        lat10_60_nodam.append(lat[j])



    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        temp = flatData(data)
        vmin = np.percentile(temp, 5)
        vmax = np.percentile(temp, 95)
        if cRangeint is True:
            vmin = int(round(vmin))
            vmax = int(round(vmax))
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    if len(data.squeeze().shape) == 1:
        isGrid = False
    else:
        isGrid = True
    if bounding is None:
        bounding = [np.min(lat)-0.5, np.max(lat)+0.5,
                    np.min(lon)-0.5, np.max(lon)+0.5]

    mm = basemap.Basemap(
        llcrnrlat=bounding[0],
        urcrnrlat=bounding[1],
        llcrnrlon=bounding[2],
        urcrnrlon=bounding[3],
        projection=prj,
        resolution='c',
        ax=ax)
    mm.drawcoastlines()
    mm.drawstates(linestyle='dashed')
    mm.drawcountries(linewidth=1.0, linestyle='-.')
    x, y = mm(lon, lat)
    x99_dam , y99_dam = mm(lon99_dam, lat99_dam)
    x60_99_dam, y60_99_dam = mm(lon60_99_dam, lat60_99_dam)
    x10_60_dam, y10_60_dam = mm(lon10_60_dam, lat10_60_dam)
    x99_nodam, y99_nodam = mm(lon99_nodam, lat99_nodam)
    x60_99_nodam, y60_99_nodam = mm(lon60_99_nodam, lat60_99_nodam)
    x10_60_nodam, y10_60_nodam = mm(lon10_60_nodam, lat10_60_nodam)
    xPUB, yPUB = mm(lon_PUB, lat_PUB)
    if isGrid is True:
        xx, yy = np.meshgrid(x, y)
        cs = mm.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)
        # cs = mm.imshow(
        #     np.flipud(data),
        #     cmap=plt.cm.jet(np.arange(0, 1, 0.1)),
        #     vmin=vmin,
        #     vmax=vmax,
        #     extent=[x[0], x[-1], y[0], y[-1]])
    else:
        #markersize= np.percentile(data,10)
        cs = mm.scatter(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax)
        cs10_60_dam = mm.scatter(
            x10_60_dam, y10_60_dam, c=data10_60_dam, s=40, marker='o', cmap=CMAP, vmin=vmin, vmax=vmax)
        cs10_60_nodam = mm.scatter(
            x10_60_nodam, y10_60_nodam, c=data10_60_nodam, s=40, marker='s', cmap=CMAP, vmin=vmin, vmax=vmax)
        cs99_dam = mm.scatter(
            x99_dam, y99_dam, c=data99_dam, s=160, marker='o', cmap=CMAP, vmin=vmin, vmax=vmax)
        cs99_nodam = mm.scatter(
            x99_nodam, y99_nodam, c=data99_nodam, s=160, marker='s', cmap=CMAP, vmin=vmin, vmax=vmax)
        cs60_99_dam = mm.scatter(
            x60_99_dam, y60_99_dam, c=data60_99_dam, s=100, marker='o', cmap=CMAP, vmin=vmin, vmax=vmax)
        cs60_99_nodam = mm.scatter(
            x60_99_nodam, y60_99_nodam, c=data60_99_nodam, s=100, marker='s', cmap=CMAP, vmin=vmin, vmax=vmax)
        csPUB = mm.scatter(
            xPUB, yPUB, c=dataPUB, s=160, marker='*', cmap=CMAP, vmin=vmin, vmax=vmax)
    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                mm.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            mm.plot(x, y, color='r', linewidth=3)
    if pts is not None:
        mm.plot(pts[1], pts[0], 'k*', markersize=4)
        npt = len(pts[0])
        for k in range(npt):
            plt.text(
                pts[1][k],
                pts[0][k],
                string.ascii_uppercase[k],
                fontsize=18)
    if clbar is True:
        cbar = mm.colorbar(cs10_60_dam, pad='1%')    #cs10_60_dam
        cbar.ax.tick_params(labelsize=20)
    if title is not None:
        ax.set_title(title, fontsize=26)

    # produce a legend with a cross section of sizes from the scatter
    #
    #ax.legend(handles, labels, loc="lower right")
    legend1 = ax.legend(handles=[cs99_dam, cs99_nodam, cs60_99_dam, cs60_99_nodam, cs10_60_dam, cs10_60_nodam],
              labels=['without major dam', 'with Major dam'],
              loc='lower left',
              fontsize=17)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=[cs99_dam, cs99_nodam, cs60_99_dam, cs60_99_nodam, cs10_60_dam, cs10_60_nodam, csPUB],
              labels=['(p>99)', '(p>99)', '(60<p<99)', '(60<p<99)', '(10<p<60)', '(10<p<60)', 'PUB'],
              loc='lower right',
              fontsize=17)
    ax.add_artist(legend2)
    if dataPUB is not None:
        legend3 = ax.legend(handles=[csPUB],
                            labels=['PUB'],
                            loc=(0.008, 0.16),
                            fontsize=17)
    if ax is None:
        return fig, ax, mm
    else:
        return mm, cs