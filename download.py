from sklearn.datasets import fetch_openml


# ----- Use this script to download the MNIST dataset -----
mnist = fetch_openml("mnist_784", version=1, as_frame=True, parser="auto")

# Only take the first 2000 rows
rows = 2000
df = mnist.frame[:rows]

# Save as a csv
df.to_csv(f"./data/mnist_{rows}.csv", index=False)

# Note: make sure the directory data/ exists before running the script