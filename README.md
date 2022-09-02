# Portfolio

In this repository you will find some projects of my python coding portfolio. 

1) Data Visualization

* Basic work with data (here Gapminder datasets)
    - import files (csv, xlxs - using Pandas)
    - manipulation of dataframes (using Pandas)
    - removing NaN values (Pandas)

* Data visualization: 
    - Scatter plot of correlation between fertility rate and life expectancy compared between countries (matplotlib, seaborn)
    - Animation of the scatterplot depending on the year (matplotlib, seaborn, imageio)
    - Creating for loops 
    -Data visualization with slider to select the depicted year (plotly)

2) Time series - future temperature predictions

* Get and clean temperature data from www.ecad.eu (chosen location - Riga, Latvia)
* Build a baseline model - model the trend and seasonality
* Plot and inspect the different components of a time series
* Model time dependence of the remainder using an autoregressive (AR) model
* Compare AR and ARIMA models
* Compare the result of the AR model and facebook prophet

Key tools:
* pandas 
* sklearn.linear_model - LinearRegression
* matplotlib
* statsmodels.graphics.tsaplots - autocorrelation and partial autocorrelation plotting
* statsmodels.tsa.arima.model - Autoregressive Integrated Moving Average (ARIMA) Model
* Facebook Prophet - timeseries predictions

3) Virtual Assistant for Alzheimer's disease diagnosis

Virtual assistant comprises two programs. First is patient metric based diagnosis prediction using logistic regression model. Second is MRI image classification with convolutional neuronal network (CNN). 

You can find the data exploration and model training in the Building models folder, and the front-end app development in Streamlit app foler. 

Key tools used within this project:
* Pandas for data exploration and user input formating 
* Matplotlip, Seaborn, Plotly for data visualization 
* Sklearn Logistic regression model for diagnosis predictions 
* TensorFlow/Keras for building convolutional neuronal network for image classifier 
* Streamlit for front-end app developemnt

