#Example for pull request tutorial.

# Predict the Market - Fluid AI Test

## Your Task

We have provided you multiple market datasets from NSE (National Stock Exchange), Bitcoin (BTC/USD), Nasdaq Composite (US Markets).

Your goal is to build an algorithm / model which is able to predict the market as accurately as possible (exciting!)
You can choose the market of your choice from above to predict for or you can predict for multiple markets as well (The choice is yours depending on which one excites you most)

You also have freedom to choose the time period you predict forward for (you know your model best, so choose a time period that shows your models performance best)
That means anything from next day predictions, to next month, to next year!

You can choose any libraries of your choice and any language of your choice (Python preferred) 

The aim is to predict the future, have fun and make money while we do it :)

<br />

## Your Performance Evaluation 

Your performance is evaluated based on the out of time results you publish and the quality of your code and the approach taken. Please also publish train and test performance results and keep an out of time set which is never shown or used by the model at the time of training. 

You should also have a predict() function in a separate .py file in your pull request. 
The function should have 1 parameter:
1) Dataframe (with the same columns as the training data) of the latest market data available (This is needed in case your model needs certain past data to predict future data). On top of the function please have a comment of how many days past data is needed for the model to make predictions.

The function should return:
1) A Dataframe of forward predictions (for as many days forward as your model supports. Eg. If your model supports 30 days forward, then an output of all 30 days prediction. The dataframe should have the columns date and price prediction

<br />

## Your Submission (How do I submit)

You submit your final code by creating a fork of this project and submitting it as a pull request to the main branch of this repository. You can follow an example [Here](https://jarv.is/notes/how-to-pull-request-fork-github/)
Also see a sample pull request [here](https://github.com/Fluid-AI/marketprophecy/pull/2)

Put your name as the pull request heading and in the details you can put your Email ID and as an additional optional item a base64 encoding of your mobile number. eg. (911111000 becomes OTExMTExMDAw). In case you are not comfortable putting your encoded mobile number you can just put in your email and name. 
In the pull request please have a readme file called results.md with published results. 

<br />

## Questions and Issues?

In case you have any questions or issues just simply put it up as an issue in the github issues section [here](https://github.com/Fluid-AI/marketprophecy/issues)

<br />

## Our Philosophy for this Interview 
Our philosophy for recruiting is why make it a process that doesn't add value in everyones lives! 
It doesn't need to always be a process of making people do things that doesn't help them in their futures.
Have you ever sat through hours of testing, not got selected and felt, what was the point. 

As such we have innovated in a process that ensures that even if you don't get selected, you learn something useful that can be helpful to you wherever you go. 

