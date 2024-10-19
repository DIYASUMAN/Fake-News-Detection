import pandas as pd

# Sample data for fake and real news
data = {
    "text": [
        "NASA Confirms Solar Storm Will Hit Earth Tomorrow.",
        "Local Man Wins Lottery for the Fifth Time!",
        "The president announced a new economic relief package to support small businesses.",
        "Scientists have discovered a cure for the common cold using nanotechnology.",
        "New study shows that drinking coffee reduces the risk of heart disease.",
        "Celebrity chef opens new restaurant in downtown area.",
        "The moon is about to crash into the Earth, causing mass destruction!",
        "New vaccine developed by scientists shows 95% effectiveness in preventing the flu.",
    ],
    "label": [
        0,  # Fake news
        0,  # Fake news
        1,  # Real news
        0,  # Fake news
        1,  # Real news
        1,  # Real news
        0,  # Fake news
        1,  # Real news
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv('fake_or_real_news.csv', index=False)

print("CSV file 'fake_or_real_news.csv' created successfully!")
