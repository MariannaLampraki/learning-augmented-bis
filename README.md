# Learning Augmented BIS (LABIS)
This is a project from my university, where we used a regression to make the Binary Interpolation Search even faster and more efficient (using the scikit-learn python module).

The project was about reading a file with temperature and humidity measurements (where each entry contained one timestamp and then multiple measurements) and we had to use Binary Interpolation Search to find an entry quickly when given the timestamp. Our professors encouraged us to use the scikit-learn python module to pick a regression and train it with data they had already provided.

The code file contains comments (in Greek) explaining the code. We added them when we sent the project for review, so if you know Greek you can also read them to understand the code better.

---
### Disclaimers:
#### I am not a genius on this field, so please be patient with me
###### To anyone more knowledgable about this stuff: I am quite sure that there is a regression that is more suitable for this kind of problem and would yield greater results (although, since with our current dataset the LABIS manages to predict the index of the data on first try, I feel that picking a better model would only have theoretical value and the difference in results would be so miniscule that the average human wouldn't even notice it). However, please note that this was part of a bigger project, it was an optional part, I was the only one who wanted to do it and we were so close to the deadline that I felt bad to ask my team more than 3 days to do this. Also note that I am still at University, so we have lots of projects each semester, and unfortnuately I don't have the time to research this more right now. All I am asking is, please, don't be harsh on me if you find that there were more efficient ways to tackle the problem.
#### I do not own the datasets used for testing. They were made by our professors (Mr. Makris and Mr. Sioutas), full rights to them regarding that. If they see this and decide they don't want me using these datasets in this repository, I encourage them to contact me and let me know, so I can remove them.
#### Also, the formulas used are from Mr. Tsakalidis' book, *Data Structures*, so the BIS/BIS* algorithm formulas were from there.

---

### Project Details
**Oracle Model Picked:** Decision Tree Regressor

**Data Given to the Oracle:** The sine and cosine values of the month, day (numbered in the month) and hour (in 24hour format) of the timestamps*

**Data Asked from the Oracle:** Index of the respective timestamp in the data structure

\* See [the project report](https://github.com/MariannaLampraki/learning-augmented-bis/wiki/Project-Report-in-English-(Automatically-Translated)) for a more detailed explanation as to why we picked this kind of data for the input.

---
