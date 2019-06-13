## FIT5147
## Student ID: 28685989
## Student Name: Hayoung Jung


# Load Libraries
library(syuzhet)  # For sentiments analysis
library(readr)  # Read csv file
library(dplyr)  # pipeline
library(tidyverse)  # text mining
library(ggplot2)  # graphing
library(tm)  # Text mining
library(tidyquant)  # stock market graph
library(rjson)  # reading json file
library(ggcorrplot)  # drawing correlation heatmap
library(treemap)  # TreeMap library
library(plotly)  # for additional interaction
library(keras)  # Keras deeplearning library to make prediction
#install_keras(tensorflow = "gpu") # gpu version should be used


# Set Working Directory. should set to file location
#setwd("D:/Uni/2019-1/FIT5147 Visualisation/Project")


#########################################################################################################################################
# Functions
#########################################################################################################################################

## scale data
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}

## inverse-transform
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}

# Change it to supervised (k step lags)
lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
  
}

# Calculate RMSE
RMSE <- function(m,o){
  sqrt(mean((m-o)**2))
}


# Data Split
prediction <- function(dateFrom, dateTo, feature_use, epochs, dataset){
  if (dataset == "S&P500"){
    combined <- read.csv("./combined.csv")
    snp.data <- read_csv("./s&p500.csv")
    snp.close <- snp.data[, c(1,2,3,5)]
    snp.close <- snp.close[which(snp.close$Date >= dateFrom & snp.close$Date <= dateTo),]
    as.data.frame(read_csv("./combined.csv")) %>%
      filter(Date <= dateTo & Date >= dateFrom) #  Filter by selected date
    
  }
  else{
    combined <- read.csv("./combined2.csv")
    snp.data <- read_csv("./kospi2.csv")
    snp.close <- snp.data[, c(1,2,3,5)]
    snp.close$Date <- dmy(snp.close$Date)
    snp.close <- snp.close[which(snp.close$Date >= dateFrom & snp.close$Date <= dateTo),]
    as.data.frame(read_csv("./combined2.csv")) %>%
      filter(Date <= dateTo & Date >= dateFrom) #  Filter by selected date
  }
  
  feature_use <- feature_use
  
  
  N <- nrow(combined)
  n <- round(N *0.8, digits = 0)
  train <- combined[1:n, 2:ncol(combined)]
  test  <- combined[(n+1):N, 2:ncol(combined)]
  
  
  # Features
  Scaled.f1 <- scale_data(train$Close, test$Close, c(-1, 1))
  Scaled.f2 <- scale_data(train$High, test$High, c(-1, 1))
  Scaled.f3 <- scale_data(train$Open, test$Open, c(-1, 1))
  Scaled.f4 <- scale_data(train$WTI, test$WTI, c(-1, 1))
  Scaled.f5 <- scale_data(train$Gold, test$Gold, c(-1, 1))
  Scaled.f6 <- scale_data(train$Sentiment, test$Sentiment, c(-1, 1))
  
  # Differences
  full.diff <- diff(combined$Close, differences = 1)  # Difference
  full.lags <- lag_transform(full.diff)  # Lag
  train.diff <- full.lags[1:n, ]  # Split
  test.diff  <- full.lags[(n+1):N, ]  # Split
  diff.scale <- scale_data(train.diff, test.diff, c(-1, 1))  # Scaling
  train.diff <- diff.scale$scaled_train  # Assign
  test.diff <- diff.scale$scaled_test  # Assign
  
  
  training_scaled.f1 <- Scaled.f1$scaled_train
  training_scaled.f2 <- Scaled.f2$scaled_train
  training_scaled.f3 <- Scaled.f3$scaled_train
  training_scaled.f4 <- Scaled.f4$scaled_train
  training_scaled.f5 <- Scaled.f5$scaled_train
  training_scaled.f6 <- Scaled.f6$scaled_train
  
  testing_scaled.f1 <- Scaled.f1$scaled_test
  testing_scaled.f2 <- Scaled.f2$scaled_test
  testing_scaled.f3 <- Scaled.f3$scaled_test
  testing_scaled.f4 <- Scaled.f4$scaled_test
  testing_scaled.f5 <- Scaled.f5$scaled_test
  testing_scaled.f6 <- Scaled.f6$scaled_test
  
  
  lookback <- 30
  
  X_train.f1 <- t(sapply(1:(length(training_scaled.f1)-lookback), function(x) training_scaled.f1[x:(x+lookback -1)]))
  X_train.f2 <- t(sapply(1:(length(training_scaled.f2)-lookback), function(x) training_scaled.f2[x:(x+lookback -1)]))
  X_train.f3 <- t(sapply(1:(length(training_scaled.f3)-lookback), function(x) training_scaled.f3[x:(x+lookback -1)]))
  X_train.f4 <- t(sapply(1:(length(training_scaled.f4)-lookback), function(x) training_scaled.f4[x:(x+lookback -1)]))
  X_train.f5 <- t(sapply(1:(length(training_scaled.f5)-lookback), function(x) training_scaled.f5[x:(x+lookback -1)]))
  X_train.f6 <- t(sapply(1:(length(training_scaled.f6)-lookback), function(x) training_scaled.f6[x:(x+lookback -1)]))
  X_train.fd <- t(sapply(1:(length(train.diff$`x-1`)-lookback), function(x) train.diff$`x-1`[x:(x+lookback-1)])) # Lags
  
  # Predict Closing 
  #y_train <- sapply((lookback +1):(length(training_scaled.f1)), function(x) training_scaled.f1[x]) # Absolute
  y_train <- sapply((lookback+1):(length(train.diff$x)), function(x) train.diff$x[x]) # Change
  
  # Reshape the input to 3-dim
  num_features = length(feature_use)
  
  X_train <- array(X_train.f1, dim=c(nrow(X_train.f1),lookback,num_features))
  
  for (i in (1: num_features)){
    X_train[,,i] <- eval(parse(text = paste("X_train.", feature_use[i], sep = "")))  #for each i th, assign X_train.f1~fn 
  }
  
  num_samples <- dim(X_train)[1]
  num_steps <- dim(X_train)[2]
  num_features <- dim(X_train)[3]
  c(num_samples, num_steps, num_features)
  
  X_test.f1 <- t(sapply(1:(length(testing_scaled.f1)-lookback), function(x) testing_scaled.f1[x:(x+lookback -1)]))
  X_test.f2 <- t(sapply(1:(length(testing_scaled.f2)-lookback), function(x) testing_scaled.f2[x:(x+lookback -1)]))
  X_test.f3 <- t(sapply(1:(length(testing_scaled.f3)-lookback), function(x) testing_scaled.f3[x:(x+lookback -1)]))
  X_test.f4 <- t(sapply(1:(length(testing_scaled.f4)-lookback), function(x) testing_scaled.f4[x:(x+lookback -1)]))
  X_test.f5 <- t(sapply(1:(length(testing_scaled.f5)-lookback), function(x) testing_scaled.f5[x:(x+lookback -1)]))
  X_test.f6 <- t(sapply(1:(length(testing_scaled.f6)-lookback), function(x) testing_scaled.f6[x:(x+lookback -1)]))
  X_test.fd <- t(sapply(1:(length(test.diff$`x-1`)-lookback), function(x) test.diff$`x-1`[x:(x+lookback-1)]))
  
  
  # Reshape the input to 3-dim
  X_test <- array(X_test.f1, dim=c(nrow(X_test.f1),lookback, num_features))
  for (i in (1: num_features)){
    X_test[,,i] <- eval(parse(text = paste("X_test.", feature_use[i], sep = "")))
  }
  
  
  #########################################################################################################################################
  # Modelling
  #########################################################################################################################################
  
  units = 4
  batch_size = 1
  
  #es_callback <- callback_early_stopping(monitor='val_mean_absolute_error', min_delta=0, patience=2, verbose=0)
  
  model <- keras_model_sequential() %>%
    layer_lstm(units, batch_input_shape = c(batch_size, num_steps, num_features), return_sequences = TRUE, stateful = TRUE)%>%
    layer_dropout(0.25) %>%
    layer_lstm(units, input_shape=c(num_steps, num_features),  return_sequences = FALSE)%>%
    layer_dropout(0.25) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = 'adam',
    metrics = c('mae')
  )
  summary(model)
  
  history <- model %>% fit(
    X_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    #callback = list(callback_tensorboard("logs/run_c")),
    shiffle = FALSE,
    validation_split = 0.2
  )
  
  history <- plot(history, smooth = getOption("keras.plot.history.smooth", TRUE))
  
  # Prediction
  pred_train <- predict(model, X_train, batch_size = 1)
  pred_test <- predict(model, X_test, batch_size = 1)
  scaler <- Scaled.f1$scaler
  
  pred_train2 <- data.frame("time"=snp.close[(1+lookback):n, 1], # - mean(close) is used to adjust final value for each market
                            "Close"=invert_scaling(pred_train, scaler, c(-1, 1))-mean(snp.close$Close) + snp.close$Close[(1+lookback):n])
  pred_test2 <- data.frame("time"=snp.close[(n+1):(N-lookback), 1], 
                           "Close"=invert_scaling(pred_test, scaler, c(-1, 1))-mean(snp.close$Close) + snp.close$Close[(n+1):(N-lookback)])
  
  p.coh2 <- ggplot() +
    geom_line(data= pred_train2, aes(y=Close, x=Date, color = 'Train')) + 
    geom_line(data= pred_test2, aes(y=Close, x=Date, color = 'Test')) + 
    geom_line(data = snp.close, aes(y=Close, x=Date, color = 'Real')) +
    theme_classic() + 
    labs(title = "Stock Market Prediction", subtitle = "Closing Price")
  
  print("Completed")
  
  return (list(p.coh2, history))
}





#########################################################################################################################################
## Server.R ##
#########################################################################################################################################

server <- function(input, output) {
  
  ## Dataset
  # S&P500 Market
  market.data <- reactive({
    if(input$dataSet == "kospi2.csv"){
      as.data.frame(read_csv(input$dataSet, col_types = list(Date = col_datetime("%d/%m/%Y")))) %>%
        filter(Date >= input$dateRange[1] & Date <= input$dateRange[2])
    }
    else{
      as.data.frame(read_csv(input$dataSet)) %>%
        filter(Date >= input$dateRange[1] & Date <= input$dateRange[2])
    }
    })

  market.data2 <- reactive({
    if(input$dataSetP == "kospi2.csv"){
      as.data.frame(read_csv(input$dataSetP, col_types = list(Date = col_datetime("%d/%m/%Y")))) %>%
        filter(Date >= input$dateRangeP[1] & Date <= input$dateRangeP[2])
    }
    else{
      as.data.frame(read_csv(input$dataSetP)) %>%
        filter(Date >= input$dateRangeP[1] & Date <= input$dateRangeP[2])
    }
  })
  
  
  # Tree Map Dataset US
  TreeUS <- as.data.frame(read_csv("data_market_cap_s&p.csv"))
  TreeUS <- TreeUS[,c(2,3,6)]
  TreeUS <- TreeUS[complete.cases(TreeUS), ]
  
  # Tree Map Dataset KR
  TreeKR <- read.csv("./data_market_cap_kospi.csv")
  TreeKR <- TreeKR[complete.cases(TreeKR), ]
  TreeKR$Market.Cap <- as.numeric(TreeKR$Market.Cap)
  
  
  # Combined Dataset
  combined <- reactive({
    as.data.frame(read_csv("./combined.csv")) %>%
    filter(Date <= input$dateRangeP[2] & Date >= input$dateRangeP[1])
    })
  
  combined.data <- reactive({combined()[, c("Date", input$predictorData)]})
  
  
  # Kmeans Dataset
  sandp.data <- read_csv("sandp500_final.csv")
  sandp.data$date <- as.Date(sandp.data$date, "%d/%m/%Y")

  sandp.data$Size <- ifelse(sandp.data$`Market Cap` >=300000000000, "Mega Cap", 
                            ifelse(sandp.data$`Market Cap` >=10000000000, "Large Cap", "Small Cap"))
  
  # Using intersect to find ticker symbols
  traded <- intersect(sandp.data[which(sandp.data$date == '2019-04-23'),]$Ticker, 
                      sandp.data[which(sandp.data$date == '2014-04-25'),]$Ticker)
  
  # Traded in both start, end print everything on start date.
  df.s <- sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$date == '2014-04-25'),][1:472, c(2,3,4,5,7,9,13,15)]
  df.e <- sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$date == '2019-04-23'),][1:472, c(2,9,13)]
  
  df.km <-data.frame("mean.mr."=double(), "SD"=double(), "Name"=character(),
                     "Sector"=character(), "Size"=character(), stringsAsFactors=FALSE)
  
  # Loop
  for (i in (1: length(traded))){
    sname <- traded[i]
    A <- sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$Ticker == sname & sandp.data$date <= '2019-04-23'),]
    a <- Delt(sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$Ticker == sname),]$close)
    ts <- xts(A$close, A$date)
    mr <- monthlyReturn(ts)
    
    df.km_temp <- data.frame(sum(mr), "SD" = sd(mr), "Name"=c(sname), "Sector"=A$Sector[1], "Size"=A$`Size`[1])
    df.km <- rbind(df.km, df.km_temp)
    
  }
  
  df.km$Sector <- ifelse(df.km$Sector == "Healthcare", 1, 
                         ifelse(df.km$Sector == "Services", 2,
                                ifelse(df.km$Sector == "Consumer Goods", 3,
                                       ifelse(df.km$Sector == "Technology", 4, 
                                              ifelse(df.km$Sector == "Utilities", 5,
                                                     ifelse(df.km$Sector == "Financial", 6,
                                                            ifelse(df.km$Sector == " Basic Materials", 7, 8)))))))
  
  kmeansData <- reactive({
    df.km[, c(input$xcol, input$ycol)]
  })
  
  clusters <- reactive({
    kmeans(kmeansData(), input$clusters)
  })
  
  output$kmeansPlot <- renderPlot({
    palette(c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
              "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"))
    par(mar = c(4, 4, 0, 1))
    plot(kmeansData(),
         col = clusters()$cluster,
         pch = 20, cex = 2)
    points(clusters()$centers, pch = 3, cex = 2, lwd = 3)
    })
  
  
  ## Prediction
  # [1] stores predicted plot
  # [2] stores training history plot
  predicted <- reactive({
    input$do
    prediction(isolate(input$dateRangeP2[1]), isolate(input$dateRangeP2[2]), isolate(input$checkGroup), isolate(input$UserEpochs), isolate(input$dataSetMP))
  })
  
  output$ModelOutput <- renderPlot({
    predicted()[1]
  })
  
  output$ModelOutput2 <- renderPlot({
    predicted()[2]
  })
  
  
  ## Overview
  # Tree Map
  # Using if else to show two different selection
  output$treeMap <- renderPlot({
    if (input$dataset == "TreeUS"){
      treemap(TreeUS, index=c("Sector", "Company"),
              vSize="Market Cap", type="index",
              fontsize.labels=c(15,12),
              fontcolor.labels=c("white","black"),
              fontface.labels=c(2,1),
              bg.labels=c("transparent"),
              align.labels=list(
                c("center", "center"), 
                c("right", "bottom")
              ),
              overlap.labels=0.5,
              inflate.labels=F)
    }
    else{
      treemap(TreeKR, index=c("Industry", "Code"),
              vSize="Market.Cap", type="index",
              fontsize.labels=c(15,12),
              fontcolor.labels=c("white","black"),
              fontface.labels=c(2,1),
              bg.labels=c("transparent"),
              align.labels=list(
                c("center", "center"), 
                c("right", "bottom")
              ),
              overlap.labels=0.5,
              inflate.labels=F)
    }

  })
  
  
  
  ## Market Analysis
  output$linePlot <- renderPlot({
      ggplot(market.data(), aes(x = Date, y= Close, open=Open, high = High, low = Low, close = Close)) + 
      geom_candlestick() +
      labs(title = "S&P500 Candle Stick",
           subtitle = "BBands with SMA, GLM 7 Smoothing",
           y = "Closing Price", x = "") + stat_smooth(formula=y~poly(x,7), method="glm") +
      theme_light()
  })
  
  ## Predictor
  output$linePlot2 <- renderPlot({
    ggplot(market.data2(), aes(x = Date, y= Close, open=Open, high = High, low = Low, close = Close)) + 
      geom_candlestick() +
      labs(title = "S&P500 Candle Stick",
           subtitle = "BBands with SMA, GLM 7 Smoothing",
           y = "Closing Price", x = "") + stat_smooth(formula=y~poly(x,7), method="glm") +
      theme_light()
  })
  
  # Create plotly plot based on ggplot2
  output$linePlot3 <- renderPlotly({
    print(
    ggplotly(ggplot(combined.data(), aes(x=Date, y=eval(parse(text = input$predictorData)))) +  # eval(parse(text))) turns text input to variable
               geom_line() + 
               labs(title = "Price",
                    subtitle = "Predictor Graph") + 
               theme_light())
    )})
  
}


#########################################################################################################################################
## ui.R ##
#########################################################################################################################################

ui <- navbarPage("Stock Market Application",
                 tabPanel("Overview",
                          sidebarLayout(
                            mainPanel(
                              plotOutput("treeMap")
                              ),
                            sidebarPanel(
                              radioButtons("dataset", "Data",
                                           c("S&P500"="TreeUS", "KOSPI"="TreeKR"))
                              )
                            )
                 ),
                 tabPanel("Market Analysis",
                          sidebarLayout(
                            mainPanel(
                              plotOutput("linePlot"),
                              plotOutput("kmeansPlot")
                            ),
                            sidebarPanel(
                              radioButtons("dataSet", "Data",
                                           c("S&P500"="s&p500.csv", "KOSPI"="kospi2.csv")
                            ),
                              dateRangeInput("dateRange",
                                           label = "Date From: yyyy-mm-dd",
                                           start = "2010-12-31",
                                           end = "2014-12-31"),
                              selectInput('xcol', 'X Variable', 
                                          c("Monthly Return" = "sum.mr.",
                                            "Standard Deviation" = "SD",
                                            "Sector" = "Sector")),
                            selectInput('ycol', 'Y Variable', 
                                        c("Standard Deviation" = "SD",
                                          "Monthly Return" = "sum.mr.",
                                          "Sector" = "Sector")),
                              numericInput('clusters', 'K means Cluster count', 3,
                                           min = 1, max = 9)
                              
                            )
                          )
                 ),
                 tabPanel("Predictors",
                          sidebarLayout(
                            mainPanel(
                              plotOutput("linePlot2"),
                              plotlyOutput("linePlot3")
                            ),
                            sidebarPanel(
                              radioButtons("dataSetP", "Data",
                                           c("S&P500"="s&p500.csv", "KOSPI"="kospi2.csv")
                              ),
                              dateRangeInput("dateRangeP",
                                             label = "Date From: yyyy-mm-dd",
                                             start = "2010-12-31",
                                             end = "2014-12-31"),
                              selectInput('predictorData', 'Predictor', 
                                          c("Open" = "Open",
                                            "High" = "High",
                                            "Close" = "Close",
                                            "WTI" = "WTI",
                                            "Gold" = "Gold",
                                            "Sentiment" = "Sentiment"))
                            )
                          )
                 ), 
                 tabPanel("Making Prediction",
                          sidebarLayout(
                            mainPanel(
                              plotOutput("ModelOutput"),
                              plotOutput("ModelOutput2")
                            ),
                            sidebarPanel(
                              radioButtons("dataSetMP", "Data",
                                           c("S&P500"="S&P500", "KOSPI"="KOSPI")
                              ),
                              dateRangeInput("dateRangeP2",
                                             label = "Date From: yyyy-mm-dd",
                                             start = "2010-01-04",
                                             end = "2014-12-12"),
                              checkboxGroupInput("checkGroup", label = h3("Predictors"), 
                                                 choices = list("Closing Price" = "f1", "Daily High" = "f2", "Opening Price" = "f3", 
                                                                "WTI oil price" = "f4", "Gold Price" = "f5", "Sentiment Analysis" = "f6",
                                                                "Time Lag" = "fd"),
                                                 selected = c("f1","f2","f3","f4","f5","f6","fd")),
                              numericInput("UserEpochs", "Desired training Epochs", 2,
                                           min=1, max=10),
                              actionButton("do", "Run Prediction")
                            )
                          )
                 )
)

shinyApp(ui = ui, server = server)