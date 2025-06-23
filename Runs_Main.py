import subprocess

runs = 1
for _ in range(runs):
    subprocess.run(['python', 'train.py', '--dataset', 'Twitter', '--alpha', '0.35',
                    '--beta', '0.35', '--newloss', '0.3', '--svd_q', '15', '--para', '0.5'])

for _ in range(runs):
    subprocess.run(['python', 'train.py', '--dataset', 'Snippets', '--alpha', '0.05',
                    '--beta', '0.05', '--newloss', '0.9', '--svd_q', '25', '--para', '0.7'])