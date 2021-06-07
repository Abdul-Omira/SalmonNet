# SalmonNet
Abdulwahab Omira, Ismael Castro, Christopher Shell

Department of Computer Science

Stanford University
# Introduction:
For our project we are investigating if Deep Learning (DL) can be applied to improve salmonforecasting models. Salmon are an important ecological, cultural, and economic species. Salmon areconsidered to be a keystone species which means that they are the essential kog in their ecosystem(Kurlansky, 2020). As previously mentioned, salmon are an essential economic resource. Salmonfishing  contributes  over$688million  to  the  US  economy  (American seafood industry steadilyincreases its footprint, 2015). Due to its commercial significance, salmon managers have to release ayearly forecast of salmon returns to estimate the amount of salmon that can be harvested.  This iswhere our project comes into the fold. We are trying to see if we can use DL to improve these salmonforecasts. The salmon forecast is a vital tool in monitoring salmon stocks for commercial, tribal, andrecreational harvests (McCormick & Falcy, 2015). If less salmon return than the predicted amount,the fisheries may over harvest and not allow enough salmon to return to the spawning grounds. Ifmore salmon return than the predicted amount, the fisheries may under harvest the resource whichwould could cost the local economy millions due to missed fish (McCormick & Falcy, 2015). Becauseof the significance of the forecast, resource managers have tried to improve the traditional forecastingmethods, however, surprisingly to our team, DL remains a relatively unexplored method. For ourresearch,  we are going to explore if DL techniques applied to salmon forecasting can improveforecasting models. The input into our model is a time-series dataset of Chinook ("King") Salmoncounts at Bonneville Dam on the Columbia River on daily and monthly time-steps. We then usedNeural Networks (NN), Recurrent Neural Networks (RNN), Grated Recurrent Units (GRU), andLong-Short Term Memory (LSTM) models to predict the amount of salmon that would return to thedam each day or each month. For example, we use the past 180 days to predict the 181th day.

# Related work
There has been very little work conducted on using ML to forecast salmon runs and even lesswork utilizing more modern algorithms.  Zhou (2003) and McCormick & Falcy (2015) attemptedto show the effectiveness of Artificial Neural Networks (ANNs) when compared to traditionalforecasting methods which have relied heavily on standard regression (such as linear regression,sibiling regression, ridge regression, and lasso regression)(Burke et al.  2013; Hand & Haeseker,2011). However, the ANNs were not able to demonstrate significant improvement over traditionalforecasting methods. Xu et al (2020) and Hilborn et al (2020) were the next papers to conduct researchon the potential of using machine learning to improve salmon forecasting. Yet, unlike McCormick &Falcy, both papers were able to demonstrate an improvement over traditional forecasting methodsand recommend that machine learning techniques be incorporated into current models. In Hilborn etal., for example, the new models were able to improve forecasting on every river system modeledwhen compared to the traditional forecasts. Yet, no model was able to perform successfully across allriver systems and age classes (Hilborn et al., 2020). Both papers primarily focused on using non-DLarchitectures such as Random Forest and Boosted Regression Trees although Hilborn attempted toCS230: Deep Learning, Winter 2018, Stanford University, CA. (LateX template borrowed from NIPS 2017.)
use RNNs but did not publish any information about their performance (Hilborn et al., 2020; Xuet al., 2020). Clearly, one of the biggest weaknesses of the current literature is the lack of researchbeing done on using DL to predict salmon runs. Our project attempts to fill this gap in the researchand tries to gain insight on whether DL, specifically recurrent neural networks, are a viable option forsalmon forecasting.

# Dataset and Features
For our salmon abundance data we are using Adult Passage data from Columbia River DART(Data Access in Real Time) repository developed by Columbia Basin Research. This adult passagedata is a time-series data set with daily salmon counts taken from Bonneville Dam dating back to1938 (Columbia River DART, 2021). This data was combined into a multivariate input with othercorrelated covariates to salmon survival. These covariates are Bakun Upwelling index taken from45N 125W (location outlined in Burke et al (2013)), Northern Oscillation Index (NOI), North PacificGrye Osciallation (NPGO), Pacific Decdal Osciallation (PDO), Oceanic Niño Index (ONI). Theseenvironmental co-factors are known to affect fish survival in the North Pacific and play a correlatedrole in predicting salmon survival at sea (Burke et al., 2013). This data was gathered from a varietyof public data repositories and is on a monthly time-stamp dating back to 1950 (Bakun 1973; DiLorenzo et al., 2008; ERDDAP 2021; Jacox et al., 2018; Japan Meteorological Agency 2006; Mantuaet al., 1997; Newman et al 2016; PSL 2021).The Bakun Upwelling index is the standard upwelling index used in marine science dating back to the20th century. Upwelling serves as a proxy for how productive the ocean is (Bakun 1973., 1997; Jacoxet al., 2018). The NOI measures climate variability between the North Pacific High and the tropics.This is a proxy for the temperature and productivity in the entire North Pacific Basin (Schwing et al.,2002). Similarly, NPGO also servers as a proxy for temperature and the productivity in the NorthPacific (Di Lorenzo et al., 2008).  Full basin-scale indices were also used such as PDO and ONI.These were proxies for temperature and climate shifts in the Pacific Ocean (Di Lorenzo et al., 2008;Mantua et al., 1997; Newman et al., 2016).Our dataset consisted of 24,734 days or 992 months of salmon count data. This data was initiallyprepossed down to 24,369 days for our single-variable daily models and 984 months for our single-variable monthly models. The daily data was broken down on a 180 day time-stamp to predict the181st day salmon count. This was done from 1939 to 2015 for our training set and 2016 to 2020 forour test set. For the monthly models, the data was broken down to take the last 6 months of data topredict month 7 salmon count. Again, our training set was split to take all months from 1939 to 2015and our test set was split to predict on 2016 to 2020. No development set was used due to the smallernature of our dataset.For the multi-variable models, the monthly salmon data was used starting in 1950.  January 1950was chosen as the starting year for our environmental model because this is the earliest where wecould get data for all our variables. Here, we have a total of 852 examples x 6 features (Salmon count,Upwelling, NOI, NPGO, PDO, ONI). This was then split down into 792 x 6 examples for our trainingset, 54 x 6 examples for our development set, and 54 examples for our test set. We chose to use adevelopment set for these more complicated models to be able to analyze if the model is overfittingdue to the increase amount of data.

# Methods
# Baseline
To evaluate our DL algorithms, we had to develop a series of baselines on the salmon data. We usedlinear regression, lasso regression, and ridge regression to create our baseline. Although there areother methods for forecasting salmon, such as moving average or Beverton-Holt stock-recruitmentmodels, linear regression made the most sense due to nature of our data (daily and monthly time-step)(McCormick & Falcy, 2015). Furthermore, linear regression, and versions of linear regression,have been used to forecast salmon returns (Hand & Haeseker, 2011). Linear regression models thecorrelation between two variables by attempting to fit a linear equation to the data.  It follows theformula of Y = a + b*X, where X is the independent variable and Y is the dependent variable.
We also chose to incorporate lasso (L1) and ridge (L2) regression as other baselines. This gave usmore data points to test our model. Lasso regression is very similar to linear regression but adds inL1 regularization.  This works by penalizing the model to the absolute value of the magnitude ofcoefficients. This, in essence, knocks out certain coefficients and makes the model simpler which canimprove forecasts. On the other hand, ridge regression use L2 regularization. L2 regularization puts apenalty on the model equal to the square of the magnitude of the coefficients. This shrinks all thecoefficients by the same amount.

# Fully Connected Neural Network (NN)
The next model, in order of increasing complexity, is a fully connected neural network. We built a4-layer neural network for both the daily and monthly data. On the daily data, the model takes inthe last 180 days of data and predicts the number of salmon on the 181th day. On the monthly data,the model takes in the last 6 months and predicts the number of salmon that return in month 7. Thisprediction is then compared to actual number of salmon on that day and the loss is computed. Theloss function that we used is mean squared error because it is common in DL time-series problems(Bernhard, 2020; Brownlee 2018).

# Recurrent Neural Network (RNN)
We also built a Recurrent Neural Network on both the daily and monthly data. We choose to build anRNN due to its increasing popularity in time-series forecasting (Bernhard, 2020). This RNN came ina simple and deep variety. The simple network had only a single layer plus one dense single nodelayer while the deep network used 4 recurrent layers and single node dense layer. The RNN was builtin a many to one fashion taking in an input of 180 days (on the daily model) and producing a singleoutput of the 181th day. The loss function was mean squared error.

# Grated Recurrent Unit (GRU)
Going off the RNN, we also explored a Grated Recurrent Unit. GRU, like RNNs, have become morein fashion for time-series forecasting (Bernhard, 2020; Brownlee, 2018). Our GRU model was builtin both a simple and robust manner, with the simple having one layer plus one dense single nodelayer and robust having 4 GRU layers plus one dense single node layer. Our loss function was meansquared error.

# Long-Short Term Memory (LSTM)
The final model we built was a simple and deep Long-Short Term Memory on both the daily andmonthly data.   LSTM are the most powerful recurrent network and,  like other recurrent-basednetworks, have started to be used more and more for time-series applications. Both our simple anddeep LSTM are built in a many-to-one design taking X number of inputs (days, months, years) andproduceing a single Y output (number of salmon on a day, month, or year). Our simple LSTM had 1LSTM layer with one single node dense layer. Our deep layer had 4 LSTM recurrent layers. The lossfunction was mean squared error.

# Loss Function
Mean Squared Error



# Experiments/Results/Discussion
# Experiments
First and foremost, we want to make it clear that out project is focused more on the comparison acrossdifferent DL models instead of extensive hyper-parameter tuning, to get a broad idea of whetheror not DL is a viable or promising approach for improving salmon forecasting. One of the thingswe experimented with was our optimizer. We had initially started with SGD, but when we saw thatthis made our models struggle to converge their loss function, we switched to the Adam optimizer.Adam worked much better and we decided to use it in all of our models for consistency. Anotherhyperparamter we experimented with was mini batch size. For our all GRU models except our multivariable models, we used a mini batch size of 150. We used a larger mini batch size for the multivariable models of 1,000 in order to speed up the computation and allow for smoother convergence ofthe loss. We followed the same scheme for our LSTM models, excpet that our multi variable shallowLSTM provided better results when ran on a mini batch size of 2000. For our classic RNN models,we used a small mini batch size of 64 to speed up convergence, at the cost of increased oscillation ofthe loss. For the multi variable RNN models, we used a mini batch size of 100 as we found it to dealbetter with the addition of 5 more covariates. Finally, we used a mini batch size of 100 for our fully connected Neural Networks, however we did not experiment much with these as they predominantlyserve as a baseline.

# Results
<img width="644" alt="results" src="https://user-images.githubusercontent.com/32625230/121101702-f819fb80-c7c1-11eb-9800-51532a4cd09b.png">


# Evaluation Metric
Since this is a regression model, we used root mean squared error (RMSE) as our evaluation metricfor all models. At one point in the project, we were comparing our models’ results to the traditionalyearly salmon forecasts (using official forecasting data), however we quickly realized that we werenot making a fair comparison since we are using the last 180 days to make a prediction on thefollowing day, whereas the traditional forecast uses the last couple years to predict the followingyear. After this realization, we created basic linear regression baselines that included a classic linearregression model, a linear regression model with Lasso penalization, and a linear regression modelwith Ridge penalization that uses the same 180 day input to 1 day output scheme as our models inorder to have a more fair comparison.

# Qualitative Results
Overall, our models that use a daily time stamp work significantly better than those with a monthlytime stamp.  This makes sense since a daily time series significantly increases the amount of datawe have, thus making it a more apt model for DL. More specifically, our shallow RNN, LSTM, andGRU, that only take daily salmon count as input, seemed to work the best on daily time stamps. Onthe other hand, our LSTM models (all LSTM models listed in section above) seem to be performingthe worst. We hypothesize that this is due to the cyclical/seasonal nature of our time series, whichcontains long stretches of zeros. The long term correlation capabilities of the LSTM may be focusingtoo much on these zeros and doing a worse job on the actual salmon runs. Overall, our models withmulti variable input fail to add any level of skill to our established monthly predictions.  This isbecause we are overfitting the training set. Please refer to section 5.3 on row "deep multivar RNN".This model had very low train set RSME, but a much higher RMSE on the test set. If we had moretime, it would be great to do our own analysis as to which covariates add the most skill to the model,instead of relying on the analysis of previous studies to select covariates. Although there is a plethoraof evidence that our environmental covariates are useful markers for salmon runs, nobody has evertried using this data in DL models, therefore further analysis may have to be done.

# Conclusion/Future Work
There is potential that Deep Learning could be utilized to improve salmon forecasting. The simpleGRU was able to out perform the baseline on the monthly data.  However, on the daily data, ourbaseline regression methods and DL models performed about the same with Lasso Regression havingthe lowest error.  This was surprising considering that daily models were fed the largest amountof data and DL algorithms tend to perform better in big data scenarios.  We think that maybe thisoccurred due to the lack of environmental data in the daily models. In the multi-variable models, themajority of our DL models outperformed the baseline.If we had more time, we would have like to build these models further and add in the yearly time-stamp data as well. Furthermore, we would have incorporated more models to compare to our DLalgorithms. It would also be interesting to apply transfer learning to see if the model could producegood results on other river systems. Currently, our model is only trained on salmon that return to theColumbia River so it would be interesting to see if we could successfully forecast salmon that returnto the Sacramento River, for example. If this ended up being the case, then it would show that usingDL as a forecasting tool is scalable and could replace current forecasting methods.

# References
[1] \textit{American seafood industry steadily increases its footprint.} American seafood industry steadily increases its footprint | National Oceanic and Atmospheric Administration. (2018, December 18). https://www.noaa.gov/media-release/american-seafood-industry-steadily-increases-its-footprint. 

[2] Bakun, A. (1973). Coastal upwelling indices, west coast of North America. US Department of Commerce. NOAA Technical Report, NMFS SSRF-671 

[3] Bernhard, J. (2020, July 14). Predicting Stock Prices Using Deep Learning Models. Medium. https://medium.com/swlh/predicting-stock-prices-using-deep-learning-models-310b41cec90a

[4] Brownlee, J. (2020, October 20). Multivariate Time Series Forecasting with LSTMs in Keras. Machine Learning Mastery. https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/. 

[5] Brownlee, J. (2018). Deep Learning for Time Series Forecasting. 

[6] Burke, B. J., Peterson, W. T., Beckman, B. R., Morgan, C., Daly, E. A., & Litz, M. (2013). Multivariate models of adult Pacific salmon returns. PloS one, 8(1), e54134.

[7] Columbia River DART, Columbia Basin Research, University of Washington. (2021).

[8]  Di Lorenzo et al., 2008: North Pacific Gyre Oscillation links ocean climate and ecosystem change, GRL.  

[9] ERDDAP, NOAA (2021). \textit{Oscillation Indices (NOI, SOI, SOI}, \textit{Monthly, 1950 - 2020} (ERDDAP) [Data Access Form]. https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdlasNoix.html 

[10] Folland, C. K. and D. E. Parker, 1995: Correction of instrumental biases in historical sea surface temperature data. Q. J. R. Meteorol. Soc., 121, 319-367.

[11] Hand, David and Haeseker, Steve, 2011. Retrospective Analysis of Preseason Run Forecast Models for Warm Springs stock
Spring Chinook Salmon in the Deschutes River, Oregon. USFWS-Columbia River Fisheries Program Office. https://www.fws.gov/columbiariver/publications/Analysis_of_Warm_Springs_Stock_Spring_Chinook_Salmon_Forecasting_Methods.pdf

[12] Hilborn, Ray, et al. BBRSDA, 2020, Improving Preseason Forecasts with Artificial Intelligence Methods and Ecosystem Information.

[13] McCormick, J.L. and Falcy, M.R. (2015), Evaluation of non‐traditional modelling techniques for forecasting salmon returns. Fish Manag Ecol, 22: 269-278. https://doi.org/10.1111/fme.12122

[14] Jacox, M. G., C. A. Edwards, E. L. Hazen, and S. J. Bograd (2018) Coastal upwelling revisited: Ekman, Bakun, and improved upwelling indices for the U.S. west coast. Journal of Geophysical Research, doi:10.1029/2018JC014187. 

[15] Kurlansky, M., Lichatowich, J., & Gayeski, N. (2020). Salmon : A Fish, the Earth, and the History of Their Common Fate. Patagonia.

[16] Mantua, N.J., S. R. Hare, Y. Zhang, J. M. Wallace, and R. C. Francis, 1997: A Pacific Interdecadal Climate Ooscillation with Impacts on Salmon Production. Bull. Amer. Meteor. Soc., 78, 1069-1079.

[17] Newman, M., M. A. Alexander, T. R. Ault, K. M. Cobb, C. Deser, E. Di Lorenzo, N. J. Mantua, A. J. Miller, S. Minobe, H. Nakamura, N. Schneider, D. J. Vimont, A. S. Phillips, J. D. Scott, and C. A. Smith, 2016: The Pacific Decadal Oscillation, Revisited. J. Clim., DOI: 10.1175/JCLI-D-15-0508.1

[18] Humdata, OCHA (2021). \textit{Oceanic Niño Index Data, Monthly 1950 - 2017} (OCHA Services) [Datasets]. https://data.humdata.org/dataset/monthly-oceanic-nino-index-oni

[19] PSL, NOAA (2021). \textit{Oceanic Niño Index Data, Monthly 2017 - 2020} (PSL) [Data Access Form]. https://psl.noaa.gov/data/correlation/oni.data

[20] Schwing, F. B., Murphree, T., & Green, P. M. (2002). The Northern Oscillation Index (NOI): a new climate index for the northeast Pacific. Progress in oceanography, 53(2-4), 115-139.

[21] Xu, Y., Hawkshaw, M., Fu, C., Hourston, R., Patterson, D., & Chandler, P. 68. ESTIMATING FRASER RIVER SOCKEYE SALMON RUN SIZE USING A MACHINE LEARNING METHOD. State of the Physical, Biological and Selected Fishery Resources of Pacific Canadian Marine Ecosystems in 2019, 273. 

[22] Zhou, S. (2003). Application of artificial neural networks for forecasting salmon escapement. North American Journal of Fisheries Management, 23(1), 48-59.

# work citation:
During the process we found some really helpful articles that helped us a lot: 

- Multivariate Time Series Forecasting with LSTMs in Keras by Jason Brownlee. 
- Predicting Stock Prices Using Deep Learning Models by Josh Bernhard. 
