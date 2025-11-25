import csv

with open("./data/hindi/STS-B/sts-dev.tsv") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        print(row)
        break