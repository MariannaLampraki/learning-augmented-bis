# Learning Augmented BIS (LABIS)
This is a project from my university, where we used a regression to make the Binary Interpolation Search even faster and more efficient (using the scikit-learn python module).

The project was about reading a file with temperature and humidity measurements (where each entry contained one timestamp and then multiple measurements) and we had to use Binary Interpolation Search to find an entry quickly when given the timestamp. Our professors encouraged us to use the scikit-learn python module to pick a regression and train it with data they had already provided.

# Project Details
---
**Regression Picked:** [Input name here]

**Data Given to Regression:** [Some data here]*

**Data Asked By Regression:** [Some data here]

\* See [bellow](README.md#problems-we-faced) for a more detailed explanation as to why we picked this kind of data for the input.

---

After the regression predicted the index of the measurement we requested, we imputted the probable index as a starting point in our BIS algorithm. We found out that the regression model usually predicted correctly the index, so the BIS algorithm had to be run only once (to check if the point was correct). In a few cases the algorithm had to run twice or more times, but overall LABIS was by far more efficient than BIS.

# Problems We Faced
Picking the correct regression was one problem. We still are not sure if this regression was the correct one, but since this was our first machine learning project and we were nearing the deadline set by our professor, we didn't have the time to study and test all the types of regressions to find the best one. We managed to compare six of them, and [the one we picked] gave the best results for our data. When the project was completed, we decided that it was efficient enough for our needs, and we didn't inquire further.

Another problem was the data we gave the model to train it. See, at first we decided to input the raw numbers of year, month, day, hour and minutes. But, when trained with this data, LABIS did almost as well as BIS, if not worse. We realised that our model just recieves numbers, and does not realise that, for example, after `hour=23` and `minute=59` the next measurement would be `hour=0` and `minute=0`. In other words, it didn't understand the cyclical relationship of units that measure time. So we decided to use periodic functions, in order to simulate this cyclical relationship. When it worked and became much more efficient, we decided that this was the correct way to do this. This also showed us how important it is to not only pick the correct "features" (= data inputed into the regression), but also to decide _how_ to [display] them in a way that such relationships are obvious to the model we are training.
