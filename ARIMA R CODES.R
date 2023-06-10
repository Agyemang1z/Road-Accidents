## ARIMA models and Intervention Analysis
# https://www.r-bloggers.com/arima-models-and-intervention-analysis/

# Combine minitab with R studio for ARIMA graphs
library(tseries)
library(ggplot2)
library(reshape2)
library(fUnitRoots)
library(forecast)
library(sarima)
library(lmtest)
library(hwwntest)
library(nortest)
library(nortestARMA)
library(bbmle)
library(seasonal)
library(car)
library(sarima)
library(stats)
library(TTR)
library(vars)
library(prophet)
library(stats)
zx<-c(38,43,36,45,40,39,50,42,39,44,41,46)
sd(zx)


####Loading the dataset whole dataset (2013-2019)
RTA<-c(32,24,47,30,18,10,6,29,25,25,31,30,
       30,35,31,41,47,29,63,27,28,39,32,22,
       16,40,37,24,14,28,24,33,27,18,26,33,
       23,26,25,22,30,28,33,37,28,31,29,30,
       27,43,24,37,30,36,31,37,36,50,65,47,
       40,21,53,34,45,52,26,41,36,42,62,35,
       38,43,36,45,40,39,50,42,39,44,41,46)


#### DATA GROUPED BY MONTHS
Jan<-mean(c(32,30,16,23,27,40,38))
Feb<-mean(c(24,35,40,26,43,21,43))
Mar<-mean(c(47,31,37,25,24,53,36))
Apr<-mean(c(30,41,24,22,37,34,45))
May<-mean(c(18,47,14,30,30,45,40))
Jun<-mean(c(10,29,28,28,36,52,39))
Jul<-mean(c(6,63,54,33,31,26,50))
Aug<-mean(c(29,27,33,37,37,41,42))
Sep<-mean(c(25,28,27,28,36,36,39))
Oct<-mean(c(25,39,18,31,50,42,44))
Nov<-mean(c(31,32,26,29,65,62,41))
Dec<-mean(c(30,22,33,30,47,35,46))


# Boxplot
Months<-c(rep("Jan",7),rep("Feb",7),rep("Mar",7),rep("Apr",7),rep("May",7),
          rep("Jun",7),rep("Jul",7),rep("Aug",7),rep("Sep",7),
          rep("Oct",7),rep("Nov",7),rep("Dec",7))
length(Months)
Frequency <-c(32,30,16,23,27,40,38,24,35,40,26,43,21,43,47,31,37,25,24,53,36,
              30,41,24,22,37,34,45,18,47,14,30,30,45,40,10,29,28,28,36,52,39,
              6,63,54,33,31,26,50,29,27,33,37,37,41,42,25,28,27,28,36,36,39,
              25,39,18,31,50,42,44,31,32,26,29,65,62,41,30,22,33,30,47,35,46)
Frequency <- c(Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec)
data <- data.frame(Months, Frequency)
ordered_months<- factor(data$Months, levels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

# Boxplot
boxplot(Frequency~ordered_months, data = data, ylab = "Monthly number of road accidents", 
        xlab = "Months (2013-2019)",add = TRUE,drop = TRUE)


#### DATA GROUPED BY QUARTERS
Q1<-mean(c(32,24,47,30,35,31,16,40,37,23,26,25,27,43,24,40,21,53,38,43,36))
Q2<-mean(c(30,18,10,41,47,29,24,14,28,22,30,28,37,30,36,34,45,52,45,40,39))
Q3<-mean(c(6,29,25,63,27,28,24,33,27,33,37,28,31,37,36,26,41,36,50,42,39))
Q4<-mean(c(25,31,30,39,32,22,18,26,33,31,29,30,50,65,47,42,62,35,44,41,46))


summary(RTA)
#make the data a time series data
RTATseries<-ts(RTA, frequency=12, start=c(2013,1))
class(RTATseries)
seasonplot(RTATseries, col=rainbow(7), year.labels=TRUE,season.labels=TRUE,main="",
           xlab = NULL, ylab = "Road Accidents",)
?seasonplot

## Plotting the time series of Accident data
autoplot(RTATseries, col="blue", xlab = "Year", ylab = "Frequency")



#### 2019 dataset
RTA2019<-c(38,43,36,45,40,39,50,42,39,44,41,46)

## Loading the dataset for building ARIMA model (2013-2018)
RoadAccident<-c(32,24,47,30,18,10,6,29,25,25,31,30,30,35,31,41,47,29,63,27,28,39,
32,22,16,40,37,24,14,28,24,33,27,18,26,33,23,26,25,22,30,28,33,37,28,31,29,30,27,43,24,37,
30,36,31,37,36,50,65,47,40,21,53,34,45,52,26,41,36,42,62,35)
length(RoadAccident)



#make the data a time series data
Tseries<-ts(RoadAccident, frequency=12, start=c(2013,1))
class(Tseries)

## Decomposing the time series
DecompTseries<-decompose(Tseries)
ts.stl<-stl(Tseries,"periodic")  # decompose the TS
ts.sa<-seasadj(ts.stl) #de-seasonalize
acf(DecompTseries$seasonal)
## Plotting the time series of Accident data
autoplot(Tseries, col="blue", xlab = "Year", ylab = "Frequency")

# both acf() and pacf() generates plots by default for Airpassengers
ACFED<- acf(Tseries) # autocorrelation
PACFED<- pacf(Tseries)  # partial autocorrelation

### test for stationarity
kpss.test(Tseries)

# Seasonal Difference
ndiffs(Tseries)  # number for seasonal difference needed

## Difference to make it stationary 
RoadAccident_seasdiff <- diff(Tseries, differences=1)  # seasonal differencing
plot(RoadAccident_seasdiff, type="l", main="Seasonally Differenced",col="blue",xlab="Year")  # still not stationary!
autoplot(RoadAccident_seasdiff)
### test for stationary using Difference series
kpss.test(RoadAccident_seasdiff)

## Test for residuals
?whitenoise.test()
whitenoise.test(Tseries)

# both acf() and pacf() generates plots by default for Difference series
ACFSEA<- acf(RoadAccident_seasdiff) # ACF plot
PACFSEA<- pacf(RoadAccident_seasdiff)  # PACF plot

## Difference and Stationary using difference series
stationaryTS <- diff(RoadAccident_seasdiff, differences= 1)
plot(stationaryTS, type="l", main="Differenced and Stationary",col="red")  # appears to be stationary
pacf(stationaryTS)
## use the de-seasonalize component to find the seasonal component P,Q,D of 
## Sarima model to be fitted

seasonplot(Tseries, col=rainbow(7), year.labels=TRUE
          ) # seasonal frequency set as 12 for monthly data.?seasonplot

## Arima Function to select the best model
BestArima<-auto.arima(Tseries, stepwise = FALSE, trace = TRUE)
BestArima1<-auto.arima(RoadAccident_seasdiff, stepwise = FALSE, trace = TRUE)

## Checking various ARIMA models

# ARIMA 1
fit1<-Arima(RoadAccident_seasdiff, order = c(0,0,1))
summary(fit1)
?arima

# if there is a seasonal component, then the code used is
SARIMA1<-Arima(Tseries, order = c(1,1,1), 
          seasonal = list(order = c(1,0,1), period = 12), 
          include.mean = TRUE,include.drift = FALSE)
# Coefficient test
coeftest(SARIMA1)
## Goodness of fit
summary(SARIMA1) # accuracy test

SARIMA2<-Arima(Tseries, order = c(1,1,1), 
          seasonal = list(order = c(1,0,0), period = 12), 
          include.mean = TRUE,include.drift = FALSE)

# Coefficient test
coeftest(SARIMA2)
## Goodness of fit
summary(SARIMA2) # accuracy test

SARIMA3<-Arima(Tseries, order = c(1,1,0), 
          seasonal = list(order = c(0,0,1), period = 12), 
          include.mean = TRUE,include.drift = FALSE)

# Coefficient test
coeftest(SARIMA3)
## Goodness of fit
summary(SARIMA3) # accuracy test

SARIMA4<-Arima(Tseries, order = c(0,1,1), 
          seasonal = list(order = c(1,0,0), period = 12), 
          include.mean = TRUE,include.drift = FALSE)

# Coefficient test
coeftest(SARIMA4)
## Goodness of fit
summary(SARIMA4) # accuracy test

SARIMA5<-Arima(Tseries, order = c(0,1,2), 
          seasonal = list(order = c(1,0,0), period = 12), 
          include.mean = TRUE,include.drift = FALSE)

# Coefficient test
coeftest(SARIMA5)
## Goodness of fit
summary(SARIMA5) # accuracy test

SARIMA6<-Arima(Tseries, order = c(1,1,2), 
          seasonal = list(order = c(1,0,0), period = 12), 
          include.mean = TRUE,include.drift = FALSE)

# Coefficient test
coeftest(SARIMA6)
## Goodness of fit
summary(SARIMA6) # accuracy test

SARIMA7<-Arima(Tseries, order = c(0,1,3), 
               seasonal = list(order = c(1,0,2), period = 12), 
               include.mean = TRUE,include.drift = FALSE)

# Coefficient test
coeftest(SARIMA7)
## Goodness of fit
summary(SARIMA7) # accuracy test


## Auto generate of ACF and PCF plot
RoadAccident%>%
Arima(order=c(0,1,1), seasonal=c(1,0,0))%>%
residuals() %>% ggtsdisplay()

# Model residual Analysis
checkresiduals(SARIMA4)
Box.test(SARIMA4$resid,type="Ljung-Box",lag=12) 
# test for normality of residuals (do for diff lags)
LjungBoxTest(residuals(SARIMA4), k = 2, lag.max = 20)

# Modified Ljung-Box test for larger lags (model accuracy)
Box.test(SARIMA4$resid,type="Box-Pierce",lag = 12) 
Box.test(SARIMA4$resid,type="Box-Pierce",lag=24) 
Box.test(SARIMA4$resid,type="Box-Pierce",lag=36) 
Box.test(SARIMA4$resid,type="Box-Pierce",lag=48)
Box.test(SARIMA4$resid,type="Box-Pierce",lag=60)

tsdisplay(residuals(SARIMA4))
tsdiag(SARIMA4)
?Box.test
qqPlot(SARIMA4$resid) # Informal test of normality
lillie.test(SARIMA4$resid) # Formal test of normality
pacf(SARIMA4$resid,col="red")
acf(SARIMA4$resid,col="green")


## Forecasting ARIMA for SARIMA4
Dataforecast<-forecast(SARIMA4,level=0.95, h=24) # for 12 months
plot(Dataforecast,include=24,col="red")
autoplot(Dataforecast,include=12,xlab="Year",ylab="Forecasted Road Accident Cases")

## Prediction ARIMA for FIT1
?predict
paw<-predict(SARIMA4,n.ahead = 24)



######## #Prediction errors for SARIMA
Actual<-c(38,43,36,45,40,39,50,42,39,44,41,46)
Predicted<-c(43,47,40,44,42,40,46,43,44,42,38,44)

#Prediction errors
MAE=function(Actual,Predicted){mean(abs(Actual-Predicted))}
MAE(Actual,Predicted)
MAPE=function(Actual,Predicted){mean(abs((Actual-Predicted)/Actual)*100)}
MAPE(Actual,Predicted)
(MSE1=mean((Predicted-Actual)^2))
sqrt(MSE1)



######FACEBOOK PROPHET MODEL
# Prepare the data
data1 <- c(32,24,47,30,18,10,6,29,25,25,31,30,30,35,31,41,47,29,63,27,28,39,
          32,22,16,40,37,24,14,28,24,33,27,18,26,33,23,26,25,22,30,28,33,37,28,31,29,30,27,43,24,37,
          30,36,31,37,36,50,65,47,40,21,53,34,45,52,26,41,36,42,62,35)
history <- data.frame(ds = seq(as.Date('2013-01-01'), as.Date('2018-12-01'), by = 'm'),y=data1)
Holidays=c('2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01',
           '2018-01-01','2013-03-01','2014-03-01','2015-03-01','2016-03-01',
           '2017-03-01','2018-03-01','2013-04-01','2014-04-01','2015-04-01',
           '2016-04-01','2017-04-01','2018-04-01','2013-05-01','2014-05-01',
           '2015-05-01','2016-05-01','2017-05-01','2018-05-01','2013-12-01',
           '2014-12-01','2015-12-01','2016-12-01','2017-12-01','2018-12-01')
length(Holidays)
m <- prophet(history,changepoints = Holidays,n.changepoints = 30,
  weekly.seasonality = "TRUE",daily.seasonality = "TRUE",yearly.seasonality = "TRUE")

future <- make_future_dataframe(m, periods = 12, freq = "month")  # Forecast for 12 months
forecast <- predict(m, future)
dyplot.prophet(m, forecast)


######## #Prediction errors for FACEBOOK PROPHET
Actual<-c(38,43,36,45,40,39,50,42,39,44,41,46)
Predicted<-c(38,43,55,35,33,41,38,42,37,46,57,38)

#Prediction errors
MAE=function(Actual,Predicted){mean(abs(Actual-Predicted))}
MAE(Actual,Predicted)
MAPE=function(Actual,Predicted){mean(abs((Actual-Predicted)/Actual)*100)}
MAPE(Actual,Predicted)
(MSE1=mean((Predicted-Actual)^2))
sqrt(MSE1)

