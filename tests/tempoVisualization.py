import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sample_data = {
    'tempo': [120, 130, 125, 140, 150, 160, 170, 180, 190, 200,]


}
df = pd.DataFrame(sample_data)
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['tempo'], marker='o', linestyle='-', color='blue')
plt.title('Tempo Visualization Over Time')
plt.xlabel('Time')
plt.ylabel('Tempo (BPM)')
plt.show()