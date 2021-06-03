# SalmonNet

# Introduction:

For our project we are investigating if Deep Learning (DL) can be applied to improve salmonforecasting models. Salmon are an important ecological, cultural, and economic species. Salmon areconsidered to be a keystone species which means that they are the essential kog in their ecosystem(Kurlansky, 2020). As previously mentioned, salmon are an essential economic resource. Salmonfishing  contributes  over$688million  to  the  US  economy  (American seafood industry steadilyincreases its footprint, 2015). Due to its commercial significance, salmon managers have to release ayearly forecast of salmon returns to estimate the amount of salmon that can be harvested.  This iswhere our project comes into the fold. We are trying to see if we can use DL to improve these salmonforecasts. The salmon forecast is a vital tool in monitoring salmon stocks for commercial, tribal, andrecreational harvests (McCormick & Falcy, 2015). If less salmon return than the predicted amount,the fisheries may over harvest and not allow enough salmon to return to the spawning grounds. Ifmore salmon return than the predicted amount, the fisheries may under harvest the resource whichwould could cost the local economy millions due to missed fish (McCormick & Falcy, 2015). Becauseof the significance of the forecast, resource managers have tried to improve the traditional forecastingmethods, however, surprisingly to our team, DL remains a relatively unexplored method. For ourresearch,  we are going to explore if DL techniques applied to salmon forecasting can improveforecasting models. The input into our model is a time-series dataset of Chinook ("King") Salmoncounts at Bonneville Dam on the Columbia River on daily and monthly time-steps. We then usedNeural Networks (NN), Recurrent Neural Networks (RNN), Grated Recurrent Units (GRU), andLong-Short Term Memory (LSTM) models to predict the amount of salmon that would return to thedam each day or each month. For example, we use the past 180 days to predict the 181th day.


# work citation:
During the process we found some really helpful articles that helped us a lot: 

- Multivariate Time Series Forecasting with LSTMs in Keras by Jason Brownlee. 
- Predicting Stock Prices Using Deep Learning Models by Josh Bernhard. 
