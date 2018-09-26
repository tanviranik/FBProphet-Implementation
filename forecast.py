import pandas as pd
import numpy as np
from fbprophet import Prophet
import datetime
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')

#Reading csv file of monthly sales data
sales_df = pd.read_csv('Product Weekly Sales Trend-31Aug2018.csv', index_col='Order Date', parse_dates=True)
sales_df = sales_df[['Invoice Amount']]
print("Orginal Sales Data:")
print(sales_df.head(5))

print('After Cleaning Invoice Amount Dataset')
sales_df['Invoice Amount'] = sales_df['Invoice Amount'].str.replace("$","")
sales_df['Invoice Amount'] = sales_df['Invoice Amount'].str.replace(",","")
sales_df['Invoice Amount'] = sales_df['Invoice Amount'].str.replace("(","")
sales_df['Invoice Amount'] = sales_df['Invoice Amount'].str.replace(")","")
sales_df['Invoice Amount'] = sales_df['Invoice Amount'].astype(float)
print(sales_df.head(5))


#print('Dataset after aggregating: daywise')
#df = sales_df.resample('86400S').sum()  #Daily data aggregated
#print(df.head(100))
#print(sales_df.resample('MS').sum())  #Monthly data aggregated


#print("After Index Reset of Sales Data:")
#df = df.reset_index()
#print(df.head())

##Column rename to get ready for prophet implementation
#df = df.rename(columns={'Order Date':'ds', 'Invoice Amount':'y'})
#df.set_index('ds').y.plot()
##plt.show()
#df['y'] = np.log(df['y'])
#df.set_index('ds').y.plot()
##plt.show()

#model = Prophet()
#model.fit(df)
#future = model.make_future_dataframe(periods=365)
#forecast = model.predict(future)
#print("Forecast Done. Forecast Dataframe:")
#forecast.tail()
#model.plot(forecast)
#model.plot(forecast).savefig('ForecastDailySales_bob.png')
#model.plot_components(forecast)
#model.plot_components(forecast).savefig('ForecastDailySalesComponents_bob.png')
#plt.show()

#df.set_index('ds', inplace=True)
#print(df.head(10))
#print("Forecast dataframe with datetime index to join with orginal data.")
#forecast.set_index('ds', inplace=True)
#print(forecast.head(10))
#viz_df = sales_df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
#print(viz_df.head())


#viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])
#viz_df['yhat_lower_rescaled'] = np.exp(viz_df['yhat_lower'])
#viz_df['yhat_upper_rescaled'] = np.exp(viz_df['yhat_upper'])
#print("After removing log scale to get back orginal data")
#print(viz_df.head())
#viz_df[['Invoice Amount', 'yhat_rescaled']].plot()
#plt.show()
#viz_df.reset_index(level=0, inplace=True)
#viz_df.to_csv("CombinedDemoDailyForecast_bob.csv", encoding='utf-8', index=False)





print('Dataset after aggregating: Monthwise')
df = sales_df.resample('MS').sum()  #Monthly data aggregated
print(df.head(5))

print("After Index Reset of Sales Data:")
df = df.reset_index()
print(df.head(5))

#Column rename to get ready for prophet implementation
df = df.rename(columns={'Order Date':'ds', 'Invoice Amount':'y'})
df.set_index('ds').y.plot()
#plt.show()
df['y'] = np.log(df['y'])
df.set_index('ds').y.plot()
print(df.head(5))
#plt.show()


print('Forecast modelling started.')
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=12, freq = 'm')
forecast = model.predict(future)
print("Forecast Done. Forecast Dataframe:")
forecast.tail()
model.plot(forecast)
model.plot(forecast).savefig('ForecastMonthlySales_bob.png')
model.plot_components(forecast)
model.plot_components(forecast).savefig('ForecastMonthlySalesComponents_bob.png')
#plt.show()

df.set_index('ds', inplace=True)
print(df.head(5))
print("Forecast dataframe with datetime index to join with orginal data.")
forecast.set_index('ds', inplace=True)
print(forecast.head(5))

print("\r\nAfter joining with orginal data:")
viz_df = sales_df.resample('MS').sum().join(forecast[['yhat', 'yhat_lower','yhat_upper', 'trend', 'trend_lower','trend_upper', 'seasonal', 'seasonal_lower', 'seasonal_upper', 'yearly', 'yearly_lower', 'yearly_upper' ]], how = 'outer')
#print(viz_df.columns)
#viz_df = viz_df.rename(columns={'':'ds'})
print(viz_df.head(5))


viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])
viz_df['yhat_lower_rescaled'] = np.exp(viz_df['yhat_lower'])
viz_df['yhat_upper_rescaled'] = np.exp(viz_df['yhat_upper'])
viz_df['trend_rescaled'] = np.exp(viz_df['trend'])
viz_df['trend_lower_rescaled'] = np.exp(viz_df['trend_lower'])
viz_df['trend_upper_rescaled'] = np.exp(viz_df['trend_upper'])
viz_df['seasonal_rescaled'] = np.exp(viz_df['seasonal'])
viz_df['seasonal_lower_rescaled'] = np.exp(viz_df['seasonal_lower'])
viz_df['seasonal_upper_rescaled'] = np.exp(viz_df['seasonal_upper'])
viz_df['yearly_rescaled'] = np.exp(viz_df['yearly'])
viz_df['yearly_lower_rescaled'] = np.exp(viz_df['yearly_lower'])
viz_df['yearly_upper_rescaled'] = np.exp(viz_df['yearly_upper'])

print("\r\nAfter removing log scale to get back orginal data")

viz_df[['Invoice Amount', 'yhat_rescaled']].plot()
#plt.show()
viz_df.reset_index(level=0, inplace=True)
print(viz_df.head(5))
viz_df.to_csv("CombinedDemoMonthlyForecast_bob.csv", encoding='utf-8', index=False)


from datetime import datetime
viz_df = viz_df.set_index('index')
print("\r\n\r\n\r\n\r\n\r\n")
print(viz_df.head(5))

#for x in range(5,42,8):
#    print(x)



#Doing cumulative sales
startDate = "2015-01-01"
#cumulativeDf = pd.DataFrame(columns=['ds', 'y'])
#cumulativeDf["ds"] = cumulativeDf['ds'].astype('datetime64')
#cumulativeDf["y"] = cumulativeDf['y'].astype('float64')
#cumulativeDf = cumulativeDf.reset_index()
#cumulativeDf = cumulativeDf.drop(['index'], axis=1)
#print(cumulativeDf)
frames = []
for x in range(0,8):
    time1 = datetime.strptime(startDate, "%Y-%m-%d")
    #next = (time1 + relativedelta(months=5)).strftime("%Y-%m-%d")
    time2 = time1 + relativedelta(months=5)
    temp = viz_df.ix[time1:time2]['Invoice Amount'].cumsum()
    
    startDate = (time1 + relativedelta(months=6)).strftime("%Y-%m-%d")
    temp = temp.reset_index()
    temp.columns = ['index', 'Cumulative Sales']
    frames.append(temp)

cumulativeDf = pd.concat(frames)
cumulativeDf = cumulativeDf.dropna()
cumulativeDf.set_index('index', inplace=True)
print(cumulativeDf)

print('--------------------------------------')
final = cumulativeDf.join(viz_df, how = 'outer')
final.reset_index(level=0, inplace=True)
print(final.head(7))

#cumulative seasonal_resclaed
final.set_index('index', inplace=True)
frames2 = []
startDate = "2015-01-01"
for x in range(0,10):
    time1 = datetime.strptime(startDate, "%Y-%m-%d")
    #next = (time1 + relativedelta(months=5)).strftime("%Y-%m-%d")
    time2 = time1 + relativedelta(months=5)
    temp = final.ix[time1:time2]['seasonal_rescaled'].cumsum()
    
    startDate = (time1 + relativedelta(months=6)).strftime("%Y-%m-%d")
    temp = temp.reset_index()
    temp.columns = ['index', 'seasonal_rescaled_cumulative']
    frames2.append(temp)

cumulativeDf = pd.concat(frames2)
cumulativeDf = cumulativeDf.dropna()
cumulativeDf.set_index('index', inplace=True)
print(cumulativeDf)

print('--------------------------------------')
final = cumulativeDf.join(final, how = 'outer')
final.reset_index(level=0, inplace=True)
print(final.head(7))


final.to_csv("CombinedDemoMonthlyForecast_bob.csv", encoding='utf-8', index=False)


    #time2 = time1 + relativedelta(months=6)
    #time3 = time2 + relativedelta(months=5)
    ##time3=datetime.strptime(next, "%Y-%m-%d")
    #print(viz_df.ix[time2:time3]['Invoice Amount'].cumsum())




#from dateutil.relativedelta import relativedelta
#import datetime
#date1 = datetime.datetime.strptime("2015-01-30", "%Y-%m-%d").strftime("%d-%m-%Y")
#print()

#today = datetime.date.today()
#print(today)
#addMonths = relativedelta(months=3)
#future = today + relativedelta(months=6)
#print(future) 
