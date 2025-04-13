
Why I choose this as my final model
    My model was inspired by the VGG architecture, but then I experimented with my own adjustment and the final model is what had the best performance; consistently high accuracy and low loss, but also good inference.
    
    My final model is a considered trade-off between accuracy and inference speed. It has a high accuracy (99.1% to 99.7%) and low loss (0.0101 to 0.0275), and I achieved these consistent good results while also keeping the model relatively simple which had much better inference time.

    The decision to use two convolutional layers before each pooling operation and use batch normalization is what really made the model start performing well. It allowed the model to understand complex features for so it can achieve high accuracy, but without making the model too deap and and have too high inference time. I avoided adding more layers or using larger kernel sizes, even though that could improved accuracy slightly, because the marginal gains I would get, would not justify the increased computational cost and potential slowdown.

    Overall, my mode is complex enough to achieve high accuracy (missing only about 0.3% to 0.9% of signs), but also streamlined enough for quick inference. The gradual increase in filter numbers allows the model to build up complexity in its understanding of the images.



What I tried:
    - Kernel sizes: I tried different combinations, from 1 x 1 to 7 x 7. Looking at the size of the pictures, I expected smaller kernels to work best, and that was also what I found through testing.

    - Filters: I tested different combinations of filters. Starting small with 32 and increasing for deeper layers and ending at 256 in the dense layer is what worked best, doubling the layers for every increase. This is in line with  common practices and empirical research. The first layers will then be able to find more details, and the deeper layers can learn more complex and abstract patterns in the pictures. To keep the filters lower (end at 256) also keeps the model computational efficient, and I still managed to get a stable high accuracy. 
    
    The reason for why this works well also has to do with forcing the model to increase it's abstract understanding when it is moving through the deeper layers. This helps to not underfit the mode - as it will understand the complex patterns, but also not giving it too much capacity so it starts to over fit - memorize the training data.

    My experimentation with higher amount of filters or the deeper layers did also not improve the results - maybe more careful tweaking coultd potentially increase accuracy slightly - but as that also means much slower inference time, I think this simpler model was overall the best. 

    The reason why I used the same filter twice, and then doubled in next layer, is based on the paper "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION" by Karen Simonyan & Andrew Zisserman from 2015, that introduced the VGG models. 

    I used a filters from 32 to 64 to 128 filters to mimic the VGG approach of gradually increasing complexity. This made my model build up from simple to complex patterns. Like VGG, I prioritized more layers which gives the model depth, over width - more filters per layer. This allows for both learing complex features but with a compact model, which gives a good inference, because it can lead to a more stable gradient flow, and lead to easier and more stable training, but also as it allows the network to make better use of each level of abstraction before moving to a more complex representation. It also can make the odel generalize better.
    
    Other benefits are:
    - By stacking two 3x3 convolutional layers before pooling (in my 32-32 and 64-64 blocks), the model has:
        - a 5x5 receptive field. This allowed my model to capture more complex patterns without the computational cost of larger filters.
        - more non-linear rectification layers, which will make the model's decision function more discriminative when classifying traffic signs.
    - By using stacked 3 x 3 filters instead of larger filters, the overall numbers of parameters in the model is smaller, making the model size and inference better.

    - Arrangement of layers: I experimented with different layer arrangements. Using two convolutional layers followed by a max-pooling layer was very effective. Stacking two layers with the same kernel size allows the model to learn basic feature first and then getting the same overview as a 5 x 5 kernel would. So it first learns local features, and then slightly more global features. Then the pooling will summarize this information. Pooling will loose some information, so by having two layers before pooling, the model can extract more information from the original input before this reduction occurs. 

    - Activation functions: While I experimented with various activation functions, including sigmoid and tanh, ReLU consistently provided the best results. It was faster to train and gave higher accuracy.

    - Optimizers: Testing different optimizers, revealed that Adam had the best results.

    - SInce we were told not to modify the main function, I instead added data augmentation directly into the get_model function. I used layers for random rotation, zoom, translation, brightness adjustment, and contrast adjustment. I choose these augmentations as this is how traffic signs often can look different in the real world. The performance decreases slightly and the training time increased significantly. I thought about keeping them, as they could make the model more robus, so it would more consistently have high accuracy if trained many times. However, I decided to remove them as the my model seems to be very robust even without data augmentation.

    - When I added more layers, I saw an improved performance until the input dimensions became too small for additional 3x3 filters. I tried adding padding and even more layers, but the performance decreased, so this seemed to be a good balance of complexity and performance.

    - I tried different activation functions, but Relu had the best result. 

    - I tried different learning rate schedules and custom initializers, but it did not improve my model

    - Regularization in my dense layer: L1/L2 regularization did not improve my model, so I removed it. The dropout layer was enough to prevent overfitting.

    - Gradient clipping did not improve the model, so it seems like the model is handling the gradients well without it. Including batch normalization and ReLU activations seems to have made the model stable enough in the gradient flow.

    - Different interpolatons: I tried different interpolation methods, and bicubic interpolation had the best balance of performance and efficiency for my model. It seemed to slightly improve model accuracy (at least more consistently) without significantly increasing the training time (since it increases loading the data). Bicubic interpolation will give more details and smoother edges, which is important when choosing class for more similar traffic signs. The small increase in computational cost when loading the data is accepted because it improved model performance.

What worked well:
    What consistently had the best results was including multiple convolutional layers, each followed by batch normalization, then add max-pooling, and dropout layers after two convolutional layers. Using two convolutional layers before each pooling layer had much better results, and it seemed like the model could learn more complex features.

    Batch normalization immediately gav a higher accuracy and lower loss, even for very simple models. This was one of the most important things I found. It also sped up training significantly. In the simpler models, the importance of batch normalization was even bigger, giving much better performance. For example, in my final model, I achieved an accuracy of 99.7 % with a loss of 0.0101 using batch normalization after each convolutional layer, compared to an accuracy of 99.0 % with a loss of 0.0359 when I trained the model without it. The final model still had high accuracy without batch normalization, but the difference was mucn bigger in the simpler models I experimented with.

    The choice of filter number also played a big role in the balance between performance and inference time. My model uses a progression of filters - 32 filters in the first two convolutional layers, then 64 in the next two, 128 in the fifth layer, and then 256 in the dense layer. This gave a good balance between performance and computational efficiency. Using 32 filters twice at the beginning let the model learn a good number of low-level features. Then doubling to 64 filters in the next two layers gave the model a chance to combine the low-level features into more complex patterns. The fifth layer with 128 filters gave the model an understanding of even more complex feature combinations, letting it understand high-level abstractions in the traffic signs. 256 neurons in the dense layer gave a good amount of feature space for the final classification decision.

    I experimented with different activation functions, and ReLU was the best. It had both the highest accuracy and was also  faster to train.

    Rescaling the images to have values between 0 and 1 gave similar results to models without it, but it made the training process more consistently have high accuracy. Therefore, I decided to include it in the final model.



What did not work well
    Some things did not perform as well as I expected. For example, adding more layers gave worse performance. More complex models with many filters or larger kernel sizes performaned better in training, but worse in validation, so it seemed to overfit. I adjusted dropout up, but removing some of the extra layers gave better results.
    
    I tried different learning rate schedules and custom initializers, but it did not improve my model. Data augmentation did not give better performance in my case, even if I tried a lot of different settings and combinations. This could have been because I would have to make the model more complex, to properly learn from the augmented data, but since my model performed so well without itq, I decided to remove it. My model already was general enough to perform well on unseen data. The dataset I had must have been good and already covered a lot of scenarios.

What I noticed:
    The best model was not the most complex one. The most important things I did was stacking two convolutional layers before pooling and dropout, and adding batch normalization betweenevery layer, which gave me a model with an accuracy of 99.7 % with a loss of 0.0101.

    Batch normalization made a big difference. As soon as I added batch normalization layers, I saw an increase in the model's accuracy and a big drop in loss, even with the simplest models. This improvement was consistent and significant, making batch normalization the most important thing I did.






