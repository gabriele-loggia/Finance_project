import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import statsmodels.api as sm
import numpy_financial  as npf
import seaborn as sns
from scipy.stats.mstats import mquantiles
warnings.filterwarnings('ignore')


#### data downloading
stocks_lst = ['INTC','QCOM','TXN','^SPX']
stocks_data_full = yf.download(stocks_lst,start='2013-4-1',end='2023-5-31', interval = '1wk')
stocks_data_ac = stocks_data_full['Adj Close']

stocks_data_ac.to_csv("stocks_data.csv")

# Loading stocks data from csv file

'''stocks_data_ac = pd.read_csv('C:\\Users\\Prova\\stocks_data.csv')
stocks_data_ac['Date'] = pd.to_datetime(stocks_data_ac['Date'])
stocks_data_ac.set_index("Date",inplace = True)'''



stocks_data_rets = np.log(stocks_data_ac/stocks_data_ac.shift()).dropna()

# CAPM ERi=Rf+b(ERm - Rf)
beta_lst = []

ERm =  stocks_data_rets['^SPX'].mean() *52 # 
Rf = 0.04 # Risk free rate
stocks_lst  = list(stocks_data_ac.columns)
for stock in stocks_lst:
    if stock != '^SPX':
        x = stocks_data_rets['^SPX']# -Rf
        x = sm.add_constant(x)
        y = stocks_data_rets[stock]
        model = sm.OLS(y, x).fit()
        beta_lst.append(list(model.params)[1])

beta_ave = sum(beta_lst)/len(beta_lst)

# Cost of Equity
CoE = Rf + (beta_ave*(ERm - Rf))
########## QUESTION 1
print('\nSOLUTION 1- Getting the wacc value and printing it out')
# WACC   =E/(E + D)*Cost of Equity + D/(E + D)*Cost of Debt*(1 - Tax Rate)

# since techy is not planning to finance the new project with debt, 
#only with equity therefore the second part of the equation becomes 0 and E/(E+D) becomes 1
# making the wacc directly equal to the Cost of equity
wacc = 1* CoE
print('\nWeighted  Average Cost of Capital = ',wacc)

print('\nPress Enter to continue')
input()    
print('\n')  


########## QUESTION 2
print('\nSOLUTION 2- Getting the NPV and IRR and printing them out')

# these are the cash flows that are given by the data.
# The -2.5 mln of initial investment and the 0.5 of gain froom the sale of the equipment at the end of the 8th year


cfs = np.array([-2.5,0.7,0.7,0.6,0.6,0.5,0.5,0.4,0.4,0.5])  
PRNPV=npf.npv(wacc,cfs)
print('\nNet Present Value=',PRNPV * 1000000 )  

print('')
PRIRR=npf.irr(cfs)
print('\nIRR value =', PRIRR)  

print('\nPress Enter to continue')
input()      
print('\n')  

########## QUESTION 3- plotting

print('\nSOLUTION 3- Plotting generated wacc values against their corresponding NPV values')

X = np.arange(0.05,0.2,0.001)

Y = np.array([npf.npv(wacc,cfs) for wacc in X])

plt.plot(X,Y)
plt.xlabel('WACC')
plt.ylabel('NPV (x 1000000 USD)')
plt.title('NPV graph')



########## QUESTION 4- Running a montecarlo simulation of generated last cashflow values 

size=2000
Cflast=np.random.triangular(0.2,0.5,0.65,size=size)

print('')
print('\nSOLUTION 4- Running a montecarlo simulation of generated last cashflow values')

n=0
npv=[] #list with npv simulation results 
irr=[] #list with irr simulation results 

while n < size:
    cfs[-1]=Cflast[n]
    npv_i=npf.npv(wacc,cfs)
    npv.append(npv_i)
    irr_i=npf.irr(cfs)
    irr.append(irr_i)
    n+=1



#converting list in arrays

npv_arr=np.array(npv) *1000000  #multiplying the NPVs by a million
irr_arr=np.array(irr)  

print("\nNPV Simulation Values array:", npv_arr)
print('\nPress Enter to continue')
input()      
print('\n') 

print("\nIRR Simulation Values array:", npv_arr)
print('\nPress Enter to continue')
input()      
print('\n') 
# calculaating the quartiles
npv_q=mquantiles(npv_arr,[0.05,0.95])
irr_q=mquantiles(irr_arr,[0.05,0.95])

print("\nNPV and IRR Simulation mean, standard deviation and quartile Values")

print('\nNPV mean = ',npv_arr.mean())
print('\nIRR mean = ',irr_arr.mean())

print('\nNPV std = ',npv_arr.std())
print('\nIRR std = ',irr_arr.std())

print('\nNPV Quantilles cfs = [0.05,0.95] =',npv_q)
print('\nIRR Quantilles cfs = [0.05,0.95] =',irr_q)

print('\nPress Enter to continue')
input()      
print('\n') 

# Plotting  NPV histogram 
sns.displot(npv,bins=20)
plt.xticks(rotation = 30)
plt.xlabel('NPV simulation values')
plt.title( 'NPV simulation histogram plot')
plt.savefig('NPV_simplot.png')

# Plotting  IRR histogram 
sns.displot(irr,bins=20)
plt.xticks(rotation = 30)
plt.xlabel('NPV simulation values')
plt.title('IRR simulation histogram plot')
plt.show()


#### Making a DataFrame of the simulation values.
sim_df = pd.DataFrame({
    'Random last values' : Cflast * 1000000,
    'NPV': npv,
    'IRR': irr
})

with pd.ExcelWriter('MonteCarlo.xlsx', engine = 'xlsxwriter') as writer:
#saving in MonteCarlo the simulation result  DF
    sim_df.to_excel(writer, sheet_name='last values sim')
    workbook  = writer.book
    worksheet = writer.sheets['last values sim']

    # Insert an image.
    worksheet.insert_image('G3', 'NPV_simplot.png')
# Since we are using the with keyword above the writer will automatically save and close
# the workbook

print("\nNPV plot saved in excel file\n")


