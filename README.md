# Language-Classifier-using-Decision-Tree

Classify given text input as English or Dutch using decision tree or boosted decision trees (AdaBoost).

## Usage
- Training

```
python train.py <training_file> <OutputFile> <learning-type>
eg. python train.py train.dat dt.pkl dt
```

- Predict

```
python predict.py <hypothesis> <test>
eg. python predict.py dt.pkl test.dat
```
