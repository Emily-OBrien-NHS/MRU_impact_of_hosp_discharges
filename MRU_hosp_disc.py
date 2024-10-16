import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

################################################################################
##################################Read in Data##################################
################################################################################
####Create the engine
cl3_engine = create_engine('mssql+pyodbc://@cl3-data/DataWarehouse?'\
                           'trusted_connection=yes&driver=ODBC+Driver+17'\
                               '+for+SQL+Server')
realtime_engine = create_engine('mssql+pyodbc://@dwrealtime/RealTimeReporting?'\
                           'trusted_connection=yes&driver=ODBC+Driver+17'\
                               '+for+SQL+Server')
####ED Data
ED_query = """
select dischargedate as DischargeDate
,DATEPART(HOUR, DischargeTime) AS DischargeHour
,count(NCAttendanceId) as EDDischarges
, sum(datediff(mi,BedRequestedDateTime, DischargeDateTime)) as BedDelayMins
,(sum(case when is4hourbreach = 'n' then 1 else 0 end)*100.0)/count(*) as FourHourPerf
,avg(datediff(mi,arrivaldatetime,dischargedatetime)) as MeanTimeInDept
from DataWarehouse.ed.vw_EDAttendance
where DischargeDate between '16-JUN-2022' and getdate()-1
and IsNewAttendance = 'y'
and DischargeDate is not NULL
group by dischargedate, DATEPART(HOUR, DischargeTime) 
order by DischargeDate desc, DATEPART(HOUR, DischargeTime) asc
"""
ED_df = pd.read_sql(ED_query, cl3_engine)
####MRU Data
MRU_query = """SET NOCOUNT ON
select FKInpatientSpellID,
		wardstaystart,
		WardStayEnd,
		datediff(minute,WardStayStart,WardStayEnd) as LosMins
into #mru
from RealTimeReporting.pcm.vw_WardStay wstay with (nolock)
where WardStayCode = 'rk950116'


----Get admissions and mean los
select cast(wardstaystart as date) as [DischargeDate], DATEPART(HOUR, wardstaystart) AS DischargeHour
		,count(*) as MRUAdmissions
		,sum(LosMins)*1.0/count(*) as MeanLoSMins
from #mru
group by cast(wardstaystart as date), DATEPART(HOUR, wardstaystart)
order by cast(wardstaystart as date) desc, DATEPART(HOUR, wardstaystart) asc
"""
MRU_df = pd.read_sql(MRU_query, realtime_engine)
####Medical Discharge data
disc_query = """SET NOCOUNT ON
select distinct
wstay.FKInpatientSpellId
		,wstay.AsclepeionPatientId
		,wstay.HospitalNumber
		,DerrifordDisWard = case when wstay.WardStayCode = 'RK950101' then prevward.WardStayDescription else wstay.WardStayDescription end
		,DerrifordDisWardCode = case when wstay.WardStayCode = 'RK950101' then prevward.WardStayCode else wstay.WardStayCode end
		,dischdttm = wstay.Discharged
into #dis --drop table #dis
from	RealTimeReporting.pcm.vw_WardStay wstay with (nolock)
left join RealTimeReporting.pcm.vw_WardStay admwstay on wstay.FKInpatientSpellId = admwstay.FKInpatientSpellId
                                    and admwstay.AdmittingWard = 'Y'	      
left join RealTimeReporting.pcm.vw_WardStay prevward on prevward.FKInpatientSpellId = wstay.FKInpatientSpellId
														and prevward.WardStayEnd = wstay.WardStayStart
left join RealTimeReporting.pcm.vw_WardStay nextward on nextward.FKInpatientSpellId = wstay.FKInpatientSpellId
														and wstay.WardStayEnd = nextward.WardStayStart
where	
cast(wstay.Discharged as date) between '16-JUN-2022' and getdate()-1---discharges from MRU opening up until yesterday
and (wstay.PatCl = 'Inpatient' or (wstay.PatCl = 'Day Case'  and DATEDIFF(dd,wstay.admitted,wstay.discharged)>0))
and		(wstay.WardStayCode in (select WardCode from [RealTimeReporting].[dbo].[covid_bed_state_wards] where WardCode <> 'RK950AAU')  --SP 1.1 6/12/2022, changed from realtimereporting.pcm.vw_BedStateWards as per NM advice & 1.3 SP 3/11/2023, removed AAU
or wstay.WardStayCode = 'RK950101')
and         admwstay.WardStayDescription not like 'ED %' --To exclude ED attendances that start a Spell	
and		(wstay.DischargingWard = 'Y' or (nextward.wardstaycode not like 'RK950%'))
---Get only discharges from a care group of Medicine
select count(FKInpatientSpellID) as MedicineDischarges
,cast(dischdttm as date) as DischargeDate
,DATEPART(HOUR, dischdttm) as DischargeHour
from #dis ips
---Use bedstate wards to find care group
inner join [RealTimeReporting].[dbo].[covid_bed_state_wards] bsw on bsw.WardCode = ips.DerrifordDisWardCode
where CareGroup = 'Medicine'
and DerrifordDisWardCode <> 'RK950116'
group by cast(dischdttm as date), DATEPART(HOUR, dischdttm)
order by cast(dischdttm as date) desc, DATEPART(HOUR, dischdttm) asc
"""
disc_df = pd.read_sql(disc_query, realtime_engine)
####Close the connection
cl3_engine.dispose()
realtime_engine.dispose()
#####Merge data togehter and sort value by date
df = disc_df.merge(ED_df, how='outer').merge(MRU_df, how='outer')
df = df.sort_values(by=['DischargeDate', 'DischargeHour'])
####remove extremes from data
df = df.loc[(~(df['BedDelayMins'] < 0))
            & (df['MeanTimeInDept'].fillna(0) < 3500)].copy()
####Ensure every hour for every date is captured in the data
index = pd.MultiIndex.from_product([df['DischargeDate'].drop_duplicates(),
                                    df['DischargeHour'].drop_duplicates()],
                                    names=['DischargeDate', 'DischargeHour'])
all_date_hours = pd.DataFrame(index = index).reset_index()
df = all_date_hours.merge(df, on=['DischargeDate', 'DischargeHour'], how='left')
################################################################################
####################################ANALYSIS####################################
################################################################################
def cross_corr(x_col, y_col, x_lab, y_lab):
    #############Get X and Y data
    X = df[x_col]
    Y = df[y_col]
    #############Define starting lag and the list of lags to do a full plot
    lag = -12
    plt_lags = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    subplots = [[0,0], [0,1], [0,2], [0,3],
                [1,0], [1,1], [1,2], [1,3],
                [2,0], [2,1], [2,2], [2,3],
                [3,0], [3,1], [3,2], [3,3]]
    lags = []
    corrs = []
    sigs = []
    #set up subplots
    fig, axs = plt.subplots(4,4, sharex=True, sharey=True)
    ##############loop over each subplot
    for _ in range(25): 
        #Add in the lag to the y data and get the correlation. If multiple
        #columns passed in, combine together into one series.
        if isinstance(Y, pd.Series):
            Y = Y.shift(lag)
        else:
            #shift by lag and combine multiple columns
            df_join = Y.join(X)
            new = []
            for col in y_col:
                temp = df_join[[x_col, col]].copy()
                temp[col] = temp[col].shift(lag)
                temp = temp.dropna()
                temp.columns = [x_col, 'combined col']
                new.append(temp)
            new = pd.concat(new)
            X = new[x_col]
            Y = new['combined col']
        #Calculate the correlation of the two series, establish if this is
        #significant or not.  Record the lag and the correlation.
        #Source:
        #https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/time-series/how-to/cross-correlation/interpret-the-results/all-statistics-and-graphs/
        corr = X.corr(Y)
        sig_test = (2/((len(X) - abs(lag))**0.5))
        significant = abs(corr) > sig_test
        t_test = ttest_ind(X, Y, nan_policy='omit')
        lags.append(lag)
        corrs.append(abs(corr))
        sigs.append(sig_test)

        #############If lag in plt_lags, plot the full scatter plot for that lag
        if lag in plt_lags:
            plot = subplots[plt_lags.index(lag)]
            #Calculate line of best fit (requires removing Nans)
            nan_idxs = Y[Y.isna()].index.union(X[X.isna()].index)
            x_lin = X.drop(nan_idxs)
            y_lin = Y.drop(nan_idxs)
            x_plt = np.linspace(X.min(), X.max())
            y_plt = (LinearRegression().fit(x_lin.values.reshape(-1,1),
                                            y_lin.values.reshape(-1,1))
                                            .predict(x_plt.reshape(-1,1)))
            #Create subplot
            axs[plot[0], plot[1]].plot(X, Y, '.')
            axs[plot[0], plot[1]].plot(x_plt, y_plt, 'r-')
            (axs[plot[0], plot[1]]
             .set_title(f'Lag={lag} hours, corr={corr:.2f},\np-value={t_test.pvalue}',
                        fontsize='x-large',
                        color='red' if significant else 'black'))
        #############Increase lag for next loop
        lag += 1
    #############Set x and y labels and save plot
    for ax in axs.flat:
        ax.set(xlabel=x_lab, ylabel=y_lab)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    #Add title, edit fig size and save.
    fig.suptitle(f'{x_lab} against {y_lab} at different time lags',
                 fontsize='xx-large', fontweight='bold')
    fig.set_figheight(15)
    fig.set_figwidth(20)
    plt.savefig(fr'.\Plots\{x_lab} against {y_lab}.png')
    plt.close()
    #############Create plot of just lags and correlations
    plt.plot(lags, corrs)
    plt.plot(lags, sigs)
    plt.xlabel('Lag (Hrs)')
    plt.ylabel('Correlation')
    plt.title(f'Correlation of {x_lab} against {y_lab} against time lags')
    plt.savefig(fr'.\Plots\Corr of {x_lab} against {y_lab}.png')
    plt.close()

#############Run the cross correlation query on the required relationships
cross_corr('MedicineDischarges', 'EDDischarges', 
           'Medicine Discharges', 'ED Discharges')
cross_corr('MedicineDischarges', 'BedDelayMins', 
           'Medicine Discharges', 'ED Bed Delays')
cross_corr('MedicineDischarges', 'FourHourPerf',
           'Medicine Discharges', 'ED 4hr Performance')
cross_corr('MedicineDischarges', 'MeanTimeInDept',
           'Medicine Discharges', 'ED Time in Dep')
cross_corr('MedicineDischarges', 'MRUAdmissions',
           'Medicine Discharges', 'MRU Admissions')
cross_corr('MedicineDischarges', 'MeanLoSMins',
           'Medicine Discharges', 'MRU Time in Dep')
cross_corr('MedicineDischarges', ['MeanTimeInDept', 'MeanLoSMins'],
           'Medicine Discharges', 'ED & MRU Time in Dep')
cross_corr('MedicineDischarges', ['EDDischarges', 'MRUAdmissions'],
           'Medicine Discharges', 'ED & MRU Discharges or Admissions')
