# rate-my-professor-vs-experience-python-ML
Results of CSE 475 Machine Learning Capstone Project

Milestone 1 (exploratory analysis): https://colab.research.google.com/drive/1qyzLGmg7xvBIZGIY9TJ-rjqEAZcsNitd?usp=sharing

Milestone 2 (model comparison): https://colab.research.google.com/drive/1OlWq_cI1X7mLplGas5fXcqwYqTmT2Gr8?usp=sharing

Milestone 3 (clustering & conclusion): https://colab.research.google.com/drive/1xpLmaA3h17OHrCo66aVkhx9uaPf9Q3Lf?usp=sharing

Overview
Previous Milestones
Initial Goals

The main goal with this dataset is to see if we can predict a professors "teaching score," which is related to their Rate My Professor data like avg rating, avg difficulty, would take again percent, and number of ratings. Importantly our main requirement for this prediction is that it utilizes a wide and diverse range of features, and doesn't just learn trivial/linear correlations between highly similar features (like i10_index and works_cited or avg_rating from would take again percent). I'm more interested if we can predict a metric like avg_rating using features from the 2 other "categories" of features, the faculty data from ASU website, and research output from OpenALEX, or vice versa.
NOTE!! Going forward, when I say "categories" I am refering to the 3 main categories of data mentioned previously, with the third being the rate my professor data. Each of these categories Pulled and parsed with completely differnt API calls and tools, thus of the "inner" features are highly correlated with each other, but not with "outer" features in different "categories".
Milestone 1 reflection

In milestone 1 we did extensive EDA on the dataset. Through this EDA I found that the dataset contained extensive problems, but most of all, there were not many significant correlations between the 3 different categories of data, as mentioned in the intial goals section. Overall we gained a good understanding of the data and how complex/messy it is given its real world nature.
Another issue with the milestone 1 professor data was the data cleaning, specifically missing value and outlier handling. There were some features with huge amounts of missing values and others with small amounts, since I was under pressure I ended up implementing a relatively basic/agressive dropping and mean imputing strategy. The outlier handling used IQR but I didn't really test it extensively. Overall, we ended up losing a ton of rows during this process from 2,080 to 1,250. For milestone 3 I will redo this process with more advanced and fine tune methods.
Milestone 2 reflection

In milestone 2 we tried 3 different machine learning frameworks, decision trees, neural networks (regression & classification), and basic cluster analysis. As I had expected from milestone 1, the models had a hard time learning and predicting rate my professor metrics based off of the other features and separate data categories. We also tried predicting other features like primary_title, bio (char count), as well as classification with binned numerical features, most of these resulted in poor accuracy too. We found that the model was quite good at learning certain features, but this was only because it was overly relying on very specific, highly correlated features, like h_index and works_count/works_cited. Overall the decision trees and neural networks didn't really satisfy my goals for the dataset.
Lastly, we did some basic cluster analysis and actually discovered some interesting patterns. We used KMeans with 3 different values for n: 2,4,6 and found that n=2 had the highest clustering scores. It found that the dataset naturally splits into 2 clusters, one cluster with the majority of datapoints separated by low research output, and the second cluster with a small minority containg high research output, interestingly RMP metrics played very little role in the splitting.
Because the other models performed poorly, and because simple clustering resulted in useful insights, we will continue with more advanced cluster analysis in milestone 3, which was approved by the professor.
Milestone 3
Goals for Milestone 3
For Milestone 3 we will be doing more advanced and extensive cluster analysis. Starting with our basic KMeans clustering seen previously, looking at silouhette/elbow plots to land on a good K factor, as well as analyzing different clusters and interpreting their results.
Next we will utilize more advanced clustering algorithms like hierarchical and DBSCAN, analyzing and interpreting their results, comparing to K-Means.
Since there are so many different features, with different scales, and combination of textua/numerical/categorical (encoded), I will also be doing PCA analysis and deciding which features contribute most to the variance, then we'll try to cut down on the number of features.
I will also redo the missing value and outlier handling with more advanced imputation methods.
Even though I wasn't able to do a traditional model deployment, I'll still go over the milestone 3 tasks as if it was a traditional model deployment like doing a mock deployment and analyzing the ethics.


Final clustering conclusion
From these 3 clustering algorithms it's clear that this dataset does not offer many opportunities for clear clustering. While it seems that 2 clusters is probably the best fit for this data, even then the intertia is high and the silhouette score is low around 0.2 - 0.25 indicating this dataset is not very naturally clusterable. Still, we were able to build somewhat interesting profiles and separate faculty into groups mostly centered around their research output.

It appears that the major challenge with this dataset is just how much each feature varies relative to each other, even with scaling this wild variance seems to be a problem. And of course there is the "curse of dimensionality" which seems to play a role in the poor clustering. While I'm sure cutting down the features could lead to better clustering, it would also lower the amount of conclusions we can draw from the data and interesting relationships present (assuming there are any). If I had more time I would probably try to learn more advanced clustering algorithms or implement more advanced dimensionality reduction methods, but even then I'm not fully convinced that this dataset is even clusterable. The data is just too noisy and there are too many non-linear relationships, not to mention the low correlation between major predictors like faculty info, research output, and rate my professor metrics.

Hypothetical Deployment Plan
Overview
Let's assume that in milestone 1&2 we actually did find signficant correlations between all 3 different categories of features (faculty info, research output, rate my professor data) and were able to build a proper model. Since this data is so complex I would have used neural networks, and would focus on both a regression model and classifier model for predicting rate my professor scores, specifically the weighted score generated by this function that is a weighted and normalized sum of avg_rating, avg_difficulty, would_take_again_percent, and num_ratings, to take into account the fact that some professors could have a low score but barely any ratings.


[ ]
df["score"] = (
    0.40 * (df["avg_rating"] / 5) +                    # Rating quality (40%)
    0.25 * (1 - (df["avg_difficulty"] - 1) / 4) +      # Course ease (25%)
    0.25 * (df["would_take_again_percent"] / 100) +    # Recommendation rate (25%)
    0.10 * (np.log1p(df["num_ratings"]) / np.log1p(df["num_ratings"].max()))  # Popularity (10%)
) # Scale to 0-100

print("Score Distribution:")
print(df["score"].describe())
Score Distribution:
count    1484.000000
mean        0.625949
std         0.183835
min         0.092952
25%         0.501705
50%         0.655915
75%         0.776485
max         0.936811
Name: score, dtype: float64
The way I envision this working would be that universities or educational institutions could use these models to predict how good of a fit a professor would be in their organization. It's important to note that just because a faculty member has a low score doesn't mean that they are a bad teacher, it might just mean their teaching style doesn't mesh well with the students or some other factors.

The university could choose to either use the model as is (which would contain a large amount of training data from different universities, although the current model is just from ASU), or they could choose to train the model with more data, either from their own institution or other more specialized institutions. They would need to use either RMP scores OR using in-house teaching data like quarterly evaluation reports, the important aspect is being able to quantify teaching impact, which is quite a hard thing to do.

Once the university is satisfied with the model, they could then use it for two things, using it for regression to predict a faculty member's "teaching score" or using it for classification to classify faculty into different teaching score bins or difficulty bins.

Mock deployment
Let's say that ASU has a new candidate coming in from another university, and they want to predict their teaching score or difficulty rating. The candidate fills in their faculty information so we now have data like how long their bio is, their expertise areas, education, primary title, etc. and we also grab their research output from openalex like their works count, cited by count, h-index, etc. We provide this data to the regression model to predict how difficult their classes will be. For now let's assume the model is quite accuracte with a test MAE of around 0.09. The model outputs a score like 0.8 which means the teacher is quite difficult, something would rate close to a 4 on rate my professor 1/5 scale. We might also want to predict an avg_rating score as well (corresponds to how highly students rated the professor in RMP), and the model outputs a 0.4, which is somewhat low.

Using this information the university might want to re-evaluate how good of a fit this candidate is, or talk with the candidate before teaching to ask them to lower their classes' difficulty. Maybe the university will want to look more into the candidates background to understand why their classes are so hard or why their rating might be a little low. The model output should always be taken with a grain of salt, and should never be the sole reason for the hiring/firing of a candidate, but I will touch on this more in the ethics section.

We can also simulate deployment for the classification model using the same candidate and their data. Insteading of predicting/classifying the candidate in terms of their avg_rating or avg_difficulty, maybe we want to see which department would be the best fit for the candidate (since one of the features in training is department and sub department). If the candidate has a background in liberal arts, we run the model with the candidate's data and it classifies them as belonging in the english department. With this data the university and candidate can now better figure out where the candidate's skills will be strongest.

Now that we have a good understanding of how the model would work in practice, it's important to analyze ethical concerns involving maintnace, compliance with legal regulations, and broader ethical implications.

Ethical considerations
Maintenance and Compliance
Given that this model will be used as a tool to influence important decisions like the hiring of a candidate or evaluating a staff member, it's extremely important that we are constantly maintaining the model and verifying its accuracy. We should always double check that a candidate's experience backs up the model's predictions, it is also very important to make sure that the training data we are using is valid and doesn't contain any inconsistencies. We also need to make sure the data processing and EDA pipeline is working as intended and not improperly imputing values or deleting massive amounts of data. This would be most relevant if a university decides to do their own specialized implementation of the model like using new features or using in-house evaluations to replace teaching scores.

There should always be a human in the loop as well, especially if the model is outputting low scores for candidates (or very high scores), if the candidate thinks there was an issue with the prediction, they should be able to contact someone involved with running the model and the necesarry steps should be taken to verify the output. We should never take the model's output as a ground truth!

Lastly we need to make sure the model is in compliance with legal regulations, both state and federal. One impotant consideration is if its even legal to use training data like certain evaluation metrics. Another important factor is if it's even legal to use this type of technology in the hiring process.

Broad ethical concerns
Lastly we can talk about the broader ethical considerations surrounding a model like this. The main one that I want to talk about is, is it even ethical to use a model like this to predict a teaching score? While I think predicting something like avg_rating or classifying what department a candidate belongs to, it gets a lot more grey when we start talking about predicting actual teaching scores. One of the main questions I struggled with during this project is what metrics do we use to predict an overall teaching score. The problem is that while something like avg_rating, difficulty, would take again percent, are all based on how much a student "liked" a teacher. It doesn't necesarilly correlate to how much that student learned or how good a teacher is at "teaching that subject." A candidate might have a really high avg_difficulty and somewhat low would take again percent, but still be a really good teacher. This is why I want to harp on how important it is we don't treat the model's responses as ground truths. In fact I think the model output should only play a small role in the hiring process, and every prediction should be verified extensively.

Another important aspect is keeping the model fair and the insitituion hosting the model accountable, not to mention the actual model makers. Often training data will contain hidden biases, I mentioned one before like a difficult teacher getting low scores but actually being a good "teacher" when it comes to sharing that knowledge. Another bias could be if a candidate has low research output, this could negatively impact their final prediction, or maybe they made a mistake when filling out their faculty info. It's difficult to say how signficant tiny mistakes or inconsistencies like this could impact the final prediction which is what makes a model like this somewhat dangerous to use in practice. With so many different variables at play, the nature of the final decision (hiring/firing can be a big impact in a candidate's life), and the blackbox nature of neural networks, it's hard to argue for the efficacy of this type of model.

Given these concerns, the developers of the model and the instituion hosting the model, should maintain the utmost level of transparency at all times. This involves keeping comprehensive logs of all the training data used, how that data was processed, scaling methods used, imputation methods, outlier handling, training hyper parameters, etc. as well as model cards. And mentioned previously there should always be a human in the loop who should always be available for contact incase any issues arise with the model. One might also consider using decision trees instead of neural networks, while we lose the ability to do regression, the classification becomes much more transparent and easier to interpret.

Final Conclusion
This was quite an insightful project for me. There were many times where I regretted choosing to collect my own data for this project since it was so complex and messy, not to mention the lack of signficant correlations. But in the end I'm still glad I did because each milestone taught me a lot. In milestone 1 I learned how to properly handle missing values or outliers and what effect this has on the final data. I learned how to interpret distribution factors like mean, variance, skewness, kurtosis, median, IQR, and how all of these things impact the final distribution and thus the model. In milestone 2 there were many times where I thought the models were making good progress, only to realize they were learning superfloous connections or that relationships I thought were obvious, turned out to be much more complex and non linear than I had expected. In milestone 3 I learned A TON about clustering and how to interpret the results, and overall what this says about the data (and my conceptions about the data, both pre and post).

My final conclusion is that predicting how good a teacher will be is much much more complex than one might think. Even though I started with around 20 features (some were dropped after processing), it turns out these were still not enough to lead to meaningful predictions. It appears the relationship between faculty info like title, deparment, expertise, research output like works count, works cited, h-index, and rate my professor metrics like avg rating and avg difficulty, is extremely complex and will require much more data and many more features to build any signficant judgements. I think this says something about the nature of learning and teaching as a whole, there isn't a good way to define a good or bad teacher, even though we have some basic metrics from student ratings in rate my professor, who is to say if this data has any real impact on how much a student learned?

From an ethical perspective, in a way, this is a good thing. If it was that easy to group teachers up into boxes and stamp them with a label, universities would have been doing this for a long time. Possibily denying well qualified candidates simply because a blackbox neural network said so. It shows that you can't read a book by its cover, and certain metrics we as people place importance on like research output, may not be important as we think they are in predicting how overall qualified a person might be for the job. I hope in the future there will be more studies done on this so one day we can finally elucidate the answer to what makes a teacher "good"!
