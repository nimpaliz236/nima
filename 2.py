import numpy as np

data = np.random.normal(size=1000)
avg = np.mean(data)
std = np.std(data)

print("avg", avg)
print("std", std)

normalized_data = (data - avg) / std
print("میانگین پس از نرمال‌سازی:", np.mean(normalized_data))
print("انحراف معیار پس از نرمال‌سازی:", np.std(normalized_data))
