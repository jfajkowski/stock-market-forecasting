import csv

currencies = {'$': 'dollars', '€': 'euros'}

with open('./data/external/currencies.csv', 'w', encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    for row in currencies.items():
        writer.writerow(row)
