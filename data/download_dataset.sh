wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar

echo "Modify 'cr' test and train splits"
# Redistribute train and test examples in cr only using 500 examples for testing.
# This is necessary to have enough training and validation examples for 512-shot experiments.
sed -i 's/1000/250/' original/cr/process.py
( cd original/cr python; python process.py )

echo "Done"
