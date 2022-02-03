## Technical notes


### Overview

Here we have a regression problem: given a mix of categorical and continuous inputs and a continuous output in the training data, the goal is to predict a continuous output (duration in seconds) for a test dataset.  The stated goal is twofold: demonstrate extent of ability and maximize test set performance.

Model ensembling is usually the best way to maximize test performance, and in this case implementing multiple models also addresses the first aim. I am going to put more emphasis on the first goal rather than the second due to time constraints, so the result may not be too accurate but should demonstrate an interesting technique.



### Approach

In my application, I mentioned that I currently work on predicting rare events using interpretable language-based deep learning methods applied to abstract strings.  In the spirit of demonstrating this here, a transformer encoder-based network will be implemented to provide a continuous output using a string input.

The approach will be to subdivide the training data into training and validation datasets, such that model parameters are updated only on this smaller training set and then model performance is evauated on the validation dataset.  The dataset is large enough such that I would not expect this split to hinder training performance, rendering techniques like k-fold cross-validation unnecessary.

Model construction proceeds as follows: first a multiple linear regression model will be fitted to the training set, and then this will be stacked and ensembled with a transformer encoder-based neural network.  



### Linear model

It is very unlikely that the true distribution is linear and thus that model is expected to have high bias, but with the benefit of low variance. An ordinary least squares linear regression is implemented in `linear_models.py`, 

The linear model yields an MSE of 4986994 and an RMS error of 2233 seconds and a mean absolute error of 742 seconds for rows that have no missing values.  We are told that the economic cost of under-prediction is twice that of overprediction, but this will not be taken into account until we train to predict residuals using the Transformer model.

We are also told that durations very far from the predictions are much worse than durations close to the predictions.  Thus we may expect an MSE error to be a better choice than L1 error for this problem, but without some quantification of 'much worse' it is difficult to be more specific than this.

The linear model above is far from perfect, and can be improved by making adding variables such as 'store_id' or normalizing predictors or adding custom cost function.  Given more time, I would implement these ideas.

We can also generate some feature interpretations from examining this model: as expected, increased estimated driving time yields increased duration, and on average each outstanding order contributes 21 minutes to duration.  We also see some interesting things here: increased busy dashers actually decreases order time (presumably as increased total dashers means a smaller wait until the next one becomes available).

===============================================================================================================
                                                   coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------------------------------
Intercept                                     2438.2339     14.895    163.696      0.000    2409.040    2467.428
C(market_id)[T.2.0]                           -493.8453     13.968    -35.356      0.000    -521.222    -466.469
C(market_id)[T.3.0]                           -194.1223     16.015    -12.121      0.000    -225.512    -162.732
C(market_id)[T.4.0]                           -438.6889     14.174    -30.951      0.000    -466.469    -410.909
C(market_id)[T.5.0]                           -278.5147     17.046    -16.339      0.000    -311.925    -245.104
C(market_id)[T.6.0]                           -335.8260     74.541     -4.505      0.000    -481.926    -189.726
total_onshift_dashers                          -19.6609      0.445    -44.225      0.000     -20.532     -18.790
total_busy_dashers                              -7.1418      0.466    -15.312      0.000      -8.056      -6.228
total_outstanding_orders                        21.3535      0.265     80.582      0.000      20.834      21.873
estimated_store_to_consumer_driving_duration     1.2551      0.020     61.837      0.000       1.215       1.295
===============================================================================================================



### Deep models

First, lets look at how a simple neural net architecture performs for duration prediction.  A four layer fully connected network is employed, using a weighted cost function.

Although a little bit of overkill for a relatively low-capacity problem (not many columns in our dataset), we can also look at how a more advanced architecture performs for predictions.  A transformer encoder is employed for self-attention on the input, followed by a 50-wide fully connected layer.  Optimization is achieved using Adaptive moment estimation, and the cost function is a modified version of MSE, in which over-estimation yields twice the unsquared error as underestimation (to match the description in the prompt).

Setting epoch size to be 1000 examples and with no hyperparameter optimization, the start of a training run is as follows:


Epoch 1 complete 
 Average error: 36163347.14 
 Elapsed time: 9.59s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 2 complete 
 Average error: 31863721.46 
 Elapsed time: 19.58s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 3 complete 
 Average error: 22657346.8 
 Elapsed time: 29.36s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 4 complete 
 Average error: 11841010.68 
 Elapsed time: 38.81s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 5 complete 
 Average error: 7621736.92 
 Elapsed time: 48.51s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 6 complete 
 Average error: 4350516.63 
 Elapsed time: 58.27s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 7 complete 
 Average error: 4631359.75 
 Elapsed time: 68.15s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 8 complete 
 Average error: 3501524.54 
 Elapsed time: 77.92s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 9 complete 
 Average error: 3431178.0 
 Elapsed time: 87.53s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch 10 complete 
 Average error: 3664480.2 
 Elapsed time: 97.3s 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




### Ensembled prediction

The approach here is to stack the models: the linear regression output is added to the inputs of the transformer model, and then this trained model is averaged with the linear model output.  

A short training run using standard MSE loss is as follows:


A full (200000 examples) training runs yield a validation RMS error of 1132.2 and a MAE of 714 seconds, or around 11.9 minutes.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validation RMS error: 1011.51 

Validation Weighted MAE: 1109.943517368283

Validation Mean Absolute Error: 714.6468591914279
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this is a slightly lower error than was achieved with the linear model (742 seconds), and half the RMS error acheived by the linear model (1011.5 vs 2233 seconds). Furthermore, this includes predictions for all rows, even those with missing values.  This means that the transformer model is able to fairly accurately predict durations for inputs with sparse data.

Note that while runs are not particularly time-consuming (around 33 minutes on my laptop), it would undoubtedly be useful to minibatch inputs to reduce the wait time.  To allow for ease of interpretation of single inputs, however, this will not be implemented.



### Feature interpretation


Particularly for complex models like neural nets, it is important to be able to understand how a model arrived at a decision (or in this case a regressive value). One way we can do this for neural nets is to calculate saliency, or the importance per input in any given prediction.  

In the past, I have had good luck with calculating input attribution using a combination of occlusion and gradientxinput.  The latter of these is now out of fashion in the deep learning field in lieu of integrated gradients, but I have found gradientxinput to nicely complementary with occlusion such that average attributions are relatively stable to differences in training example order and other stochastic factors.

As time is a factor, we will use only occlusion. The method here is to simply zero out all inputs for a given field, find the difference between the output given this modified input and the original, and then compare this difference for each field.  This is implemented in `occlusion()`.

For an example output, see `interpretation.png`




### Final notes

In many respects, the transformer model is overkill for the task of predicting dash durations given the limited data per example.  Indeed, there was a notable lack of much improvement when the transformer was employed compared to the linear model.  This is most likely the result of any real optimization on the transformer itself, but could also result from the noise added to the dataset such that the irreducible error is large enough to prevent further increases in accuracy.

It is also probable that the particular encoding method employed here is sub-optimal.  One could imagine that instead of a languange-like encoding, a one-hot vector for certain categorical elements (particularly store_id and region_id) would be superior.  This would

For data with more columns, especially qualitative columns, however, would be expected to increase the prediction accuracy significantly.










