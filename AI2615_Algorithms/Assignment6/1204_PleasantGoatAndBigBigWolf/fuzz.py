# %%
import numpy as np
import subprocess as sp

# %%

M = 30
N = 30

num_test_case = 1000

for e in range(num_test_case):
    print("Test Case", e+1)
    with open('fuzz.txt', 'w') as f:
        m = np.random.randint(1, M+1)
        n = np.random.randint(1, N+1)
        ans = np.random.randint(0, 3, size=(n, m))
        f.write(''.join([str(n), ' ', str(m), '\n']))
        for row in ans:
            for column in row:
                f.write(''.join([str(column), ' ']))
            f.write('\n')
        f.write('\n')

    srx = sp.run(
        ['srx.exe', '<fuzz.txt'],
        shell=True, capture_output=True, text=True)
    ybr = sp.run(
        ['wolfAndGoat.exe', '<fuzz.txt'],
        shell=True, capture_output=True, text=True)

    print(srx.stdout)
    print(ybr.stdout)
    print()
    if srx.stdout != ybr.stdout:
        print("!")
        break
