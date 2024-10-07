# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:12:59 2023

@author: Langley Juan Manuel
"""

import pandas as pd
from ta import trend as trd
from ta import momentum as tmom
import pandas_ta as pdta
import numpy as np

def feature_engineering(dataframe):
    df_copy = dataframe.copy()

    df_copy = pd.DataFrame()
    #df_copy.index = dataframe.index
    #df_copy["OpenDiff"] = dataframe["Close"]-dataframe["Open"]
    #df_copy["HighDiff"] = dataframe["High"]-dataframe["Close"]
    #df_copy["LowhDiff"] = dataframe["Close"]-dataframe["Low"]
    #df_copy["Changes"] = dataframe["Close"].diff() 

    

    df_copy["SMA3"] = trd.sma_indicator(dataframe["Close"], window=3)
    df_copy["SMA6"] = trd.sma_indicator(dataframe["Close"], window=6)
    df_copy["SMA14"] = trd.sma_indicator(dataframe["Close"], window=14)
    df_copy["SMA25"] = trd.sma_indicator(dataframe["Close"], window=25)
    df_copy["SMA50"] = trd.sma_indicator(dataframe["Close"], window=50)
    
    # -------------- Angulos Medias -----------------
    df_copy["ma_diff_MA6"] = df_copy["SMA6"] - df_copy["SMA6"].shift(5)
    df_copy["ANGULO_MA6"] = np.degrees(np.arctan(df_copy["ma_diff_MA6"]/5))
    
    df_copy["ma_diff_MA14"] = df_copy["SMA14"] - df_copy["SMA14"].shift(5)
    df_copy["ANGULO_MA14"] = np.degrees(np.arctan(df_copy["ma_diff_MA14"]/5))
    
    df_copy["ma_diff_MA25"] = df_copy["SMA25"] - df_copy["SMA25"].shift(8)
    df_copy["ANGULO_MA25"] = np.degrees(np.arctan(df_copy["ma_diff_MA25"]/8))
    
    df_copy["ma_diff_MA50"] = df_copy["SMA50"] - df_copy["SMA50"].shift(8)
    df_copy["ANGULO_MA50"] = np.degrees(np.arctan(df_copy["ma_diff_MA50"]/8))
    
    #df_copy.drop(columns=['SMA6'], inplace=True)
    df_copy.drop(columns=['ma_diff_MA6'], inplace=True)
    df_copy.drop(columns=['ma_diff_MA14'], inplace=True)
    df_copy.drop(columns=['ma_diff_MA25'], inplace=True)
    df_copy.drop(columns=['ma_diff_MA50'], inplace=True)
    
    # ANGULOS PRECIOS
    df_copy["ma_diff_close_8"] = pd.to_numeric(dataframe["Close"] - dataframe["Close"].shift(8), errors="coerce")
    df_copy["ANGULO_CLOSE_8"] = np.degrees(np.arctan(df_copy["ma_diff_close_8"]/8))

    df_copy["ma_diff_close_12"] = pd.to_numeric(dataframe["Close"] - dataframe["Close"].shift(12), errors="coerce")
    df_copy["ANGULO_CLOSE_12"] = np.degrees(np.arctan(df_copy["ma_diff_close_12"]/12))
    
    df_copy["ma_diff_close_16"] = pd.to_numeric(dataframe["Close"] - dataframe["Close"].shift(16), errors="coerce")
    df_copy["ANGULO_CLOSE_16"] = np.degrees(np.arctan(df_copy["ma_diff_close_16"]/16))
    
    df_copy.drop(columns=['ma_diff_close_8'], inplace=True)
    df_copy.drop(columns=['ma_diff_close_12'], inplace=True)
    df_copy.drop(columns=['ma_diff_close_16'], inplace=True)
    
    # -------------- MACD ----------------- 
    macd_indicator = trd.MACD(dataframe["Close"], window_fast=12, window_sign=9, window_slow=26)
    df_copy["MACD"] = macd_indicator.macd()
    df_copy["MACDHist"] = macd_indicator.macd_diff()
    df_copy["Trix"] = pdta.trix(dataframe["Close"], length=14)["TRIX_14_9"]
    
    
    # -------------- Diferencia entre medias -----------------
    df_copy["SMA14_dif"] = pd.to_numeric(df_copy["SMA14"] - df_copy["SMA6"], errors="coerce")
    df_copy["SMA25_dif"] = pd.to_numeric(df_copy["SMA25"] - df_copy["SMA6"], errors="coerce")
    
    
    ema25 = trd.ema_indicator(dataframe["Close"], window=25)
    df_copy["EMA25_dif"] = pd.to_numeric(ema25 - df_copy["SMA6"], errors="coerce")
    sma50 = trd.ema_indicator(dataframe["Close"], window=50)
    df_copy["EMA50_dif"] =  pd.to_numeric(sma50 - df_copy["SMA6"], errors="coerce")
    
    # ------------- ADX 14 --------------
    adx = pdta.adx(high=dataframe["High"], low=dataframe["Low"], close=dataframe["Close"], length=14)
    df_copy["ADX_14"] = adx["ADX_14"]
    df_copy["DMP_14"] = adx["DMP_14"]
    df_copy["DMN_14"] = adx["DMN_14"]
    df_copy["ADXR_14"] = (adx["ADX_14"] + adx["ADX_14"].shift(14))/2

    # ------------- Diferencia de velas ----------
    
    df_copy["Diff"] = dataframe["Close"].diff() 
    df_copy["Changes"] = dataframe["Close"].diff()
    #df_copy["Changes"] = np.where(df_copy['Diff'] > 0, 1, -1)
    df_copy.drop(columns=['Diff'], inplace=True)
    
    df_copy["mean_changes"] = trd.sma_indicator(df_copy["Changes"], window=6)
    
    # ------------- ADX 28 --------------
    """
    adx = pdta.adx(high=dataframe["High"], low=dataframe["Low"], close=dataframe["Close"], length=28)
    df_copy["ADX_28"] = adx["ADX_28"]
    df_copy["DMP_28"] = adx["DMP_28"]
    df_copy["DMN_28"] = adx["DMN_28"]
    df_copy["ADXR_28"] = (adx["ADX_28"] + adx["ADX_28"].shift(28))/2
    
    df_copy["adx_diff_adx_28"] = df_copy["ADX_28"] - df_copy["ADX_28"].shift(5)
    df_copy["ANGULO_ADX_28"] = np.degrees(np.arctan(df_copy["adx_diff_adx_28"]/5))
    df_copy.drop(columns=['adx_diff_adx_28'], inplace=True)
    """
    
    return df_copy
    