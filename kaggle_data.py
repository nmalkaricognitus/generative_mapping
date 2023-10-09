import numpy as np
import gtm

# Load the data
data = np.loadtxt('data.csv', delimiter=',')

# Train the GTM
gtm = gtm.GTM(data.shape[1])
gtm.fit(data)

# Generate new data
new_data = gtm.sample(100)

# Save the new data
np.savetxt('new_data.csv', new_data, delimiter=',')
