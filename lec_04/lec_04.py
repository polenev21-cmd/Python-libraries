import numpy as np
import pandas as pd

# Pandas
# Series
# DataFrame
# Index

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)

print(data.values)
print(type(data.index))

print(data[1])
print(data[1:3])

