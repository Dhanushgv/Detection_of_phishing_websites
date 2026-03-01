import requests
import pandas as pd

df = pd.read_csv("phishing01.csv")
urls = df[df['label'] == 1]['url']

reachable = []

for u in urls:
    if len(reachable) >= 15:
        break

    try:
        print("Checking:", u)  # so you can see progress
        r = requests.head("http://" + u, timeout=0.8, allow_redirects=True)

        if r.status_code < 400:
            reachable.append(u)

    except Exception as e:
        # show what error happened
        print("Failed:", u, "|", str(e))
        continue

print("\nReachable URLs (10 max):")
for u in reachable:
    print(u)
