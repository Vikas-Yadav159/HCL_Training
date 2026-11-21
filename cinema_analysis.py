# Project 15: Movie Theater Box Office & Concession Analysis
# Uses pandas and numpy for the analysis
# (matplotlib is only used at the end to draw the graphs)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Task 1: Load and inspect ----------
print("TASK 1: LOAD AND INSPECT")
df = pd.read_csv("15_cinema_sales.csv")
print(df.head())
print("Shape:", df.shape)
print(df.dtypes)
print(df.describe())

# ---------- Task 2: Missing values and duplicates ----------
print("\nTASK 2: MISSING VALUES AND DUPLICATES")
print(df.isnull().sum())
dup_count = df.duplicated(subset="Transaction_ID").sum()
print("Duplicate Transaction_IDs:", dup_count)
df = df.drop_duplicates(subset="Transaction_ID")
print("Shape after removing duplicates:", df.shape)

# ---------- Task 3: Fill missing Concession_Spend ----------
# I use the median because a few big spenders pull the mean up.
print("\nTASK 3: CLEAN CONCESSION SPEND")
df["Snack_Was_Missing"] = df["Concession_Spend"].isnull()
median_spend = df["Concession_Spend"].median()
df["Concession_Spend"] = df["Concession_Spend"].fillna(median_spend)
print("Filled with median:", median_spend)
print("Missing now:", df["Concession_Spend"].isnull().sum())

# ---------- Task 4: Dates and weekend flag ----------
print("\nTASK 4: DATES")
df["Show_Date"] = pd.to_datetime(df["Show_Date"])
df["Weekday"] = df["Show_Date"].dt.day_name()
df["Is_Weekend"] = np.where(df["Weekday"].isin(["Saturday", "Sunday"]), 1, 0)
df["Day_Type"] = np.where(df["Is_Weekend"] == 1, "Weekend", "Weekday")
print(df[["Show_Date", "Weekday", "Day_Type"]].head())

# ---------- Task 5: Revenue ----------
print("\nTASK 5: REVENUE")
df["Ticket_Revenue"] = df["Ticket_Quantity"] * df["Ticket_Price"]
df["Total_Spend"] = df["Ticket_Revenue"] + df["Concession_Spend"]
ticket_total = df["Ticket_Revenue"].sum()
snack_total = df["Concession_Spend"].sum()
print("Ticket revenue:", ticket_total)
print("Concession revenue:", snack_total)
print("Combined revenue:", ticket_total + snack_total)

# ---------- Task 6: Top 10 movies ----------
print("\nTASK 6: TOP 10 MOVIES")
movies = df.groupby("Movie_Title")[["Ticket_Quantity", "Ticket_Revenue"]].sum()
print("By tickets sold:")
print(movies.sort_values("Ticket_Quantity", ascending=False).head(10))
print("By ticket revenue:")
print(movies.sort_values("Ticket_Revenue", ascending=False).head(10))

# ---------- Task 7: Occupancy by screen format ----------
print("\nTASK 7: OCCUPANCY BY FORMAT")
print(df.groupby("Screen_Format")["Occupancy_Pct"].mean())
print(df.groupby("Screen_Format").mean(numeric_only=True))

# ---------- Task 8: Weekday vs weekend ----------
print("\nTASK 8: WEEKDAY VS WEEKEND")
print(df.groupby("Day_Type")[["Ticket_Quantity", "Concession_Spend"]].mean())
print(df.groupby("Day_Type")[["Ticket_Quantity", "Concession_Spend"]].sum())

# ---------- Task 9: Showtime slots ----------
print("\nTASK 9: SHOWTIME SLOTS")
print(df.groupby("Showtime_Slot")["Ticket_Quantity"].sum())
print(df.groupby("Showtime_Slot")["Occupancy_Pct"].mean())

# ---------- Task 10: Attach rate ----------
print("\nTASK 10: ATTACH RATE")
with_snack = (df["Concession_Spend"] > 0).sum()
attach_rate = with_snack / len(df) * 100
print("Attach rate (%):", round(attach_rate, 2))

# ---------- Task 11: Genre ----------
print("\nTASK 11: GENRE")
genre = df.groupby("Genre")[["Ticket_Revenue", "Concession_Spend"]].mean()
print(genre)

# ---------- Task 12: NumPy stats ----------
print("\nTASK 12: NUMPY STATS")
spend = df["Total_Spend"].to_numpy()
print("Mean:", round(np.mean(spend), 2))
print("Median:", round(np.median(spend), 2))
print("Std:", round(np.std(spend), 2))

# ---------- Task 13: IQR outliers ----------
print("\nTASK 13: LARGE BOOKINGS (IQR)")
q1 = np.percentile(df["Ticket_Quantity"], 25)
q3 = np.percentile(df["Ticket_Quantity"], 75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
print("Q1:", q1, "Q3:", q3, "Upper limit:", upper)
outliers = df[df["Ticket_Quantity"] > upper]
print("Unusually large bookings:", len(outliers))
print(outliers[["Transaction_ID", "Movie_Title", "Ticket_Quantity"]])

# ---------- Task 14: Ticket bands ----------
print("\nTASK 14: TICKET BANDS")
bins = [0, 1, 2, 5, 100]
labels = ["Solo", "Couple", "Small Group", "Large Group"]
df["Group_Band"] = pd.cut(df["Ticket_Quantity"], bins=bins, labels=labels)
print(df["Group_Band"].value_counts())
print(df.groupby("Group_Band", observed=True)["Concession_Spend"].mean())

# ---------- Task 15: Recommendations ----------
print("\nTASK 15: RECOMMENDATIONS")
occ = df.groupby("Screen_Format")["Occupancy_Pct"].mean()
slots = df.groupby("Showtime_Slot")["Ticket_Quantity"].sum()
day_snack = df.groupby("Day_Type")["Concession_Spend"].mean()

print("1.", occ.idxmax(), "has the highest occupancy. Give it more screens and prime slots.")
print("2.", occ.idxmin(), "has the lowest occupancy. Reduce its shows or offer discounts.")
print("3.", slots.idxmax(), "is the busiest slot and", slots.idxmin(), "is the quietest. Promote the quiet slot.")
print("4. Weekend snack spend is", round(day_snack["Weekend"], 2), "vs", round(day_snack["Weekday"], 2),
      "on weekdays. Offer cheaper combos on weekdays.")
print("5.", genre["Concession_Spend"].idxmax(), "movies have the highest snack spend. Promote combos with them.")

# ---------- Bonus: Graphs ----------
# All graphs are saved as PNG files in the same folder.
print("\nBONUS: CREATING GRAPHS")

# Graph 1: Top 10 movies by ticket revenue
top10 = movies.sort_values("Ticket_Revenue", ascending=False).head(10)
plt.figure(figsize=(9, 5))
plt.bar(top10.index, top10["Ticket_Revenue"], color="steelblue")
plt.title("Top 10 Movies by Ticket Revenue")
plt.xlabel("Movie")
plt.ylabel("Ticket Revenue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graph1_top_movies.png")
plt.close()

# Graph 2: Average occupancy by screen format
plt.figure(figsize=(6, 5))
plt.bar(occ.index, occ.values, color="darkorange")
plt.title("Average Occupancy by Screen Format")
plt.xlabel("Screen Format")
plt.ylabel("Occupancy (%)")
plt.tight_layout()
plt.savefig("graph2_occupancy_format.png")
plt.close()

# Graph 3: Tickets sold by showtime slot
plt.figure(figsize=(6, 5))
plt.bar(slots.index, slots.values, color="seagreen")
plt.title("Tickets Sold by Showtime Slot")
plt.xlabel("Showtime Slot")
plt.ylabel("Tickets Sold")
plt.tight_layout()
plt.savefig("graph3_showtime_slots.png")
plt.close()

# Graph 4: Weekday vs weekend average snack spend
plt.figure(figsize=(5, 5))
plt.bar(day_snack.index, day_snack.values, color=["gray", "purple"])
plt.title("Average Snack Spend: Weekday vs Weekend")
plt.ylabel("Average Concession Spend")
plt.tight_layout()
plt.savefig("graph4_weekday_weekend.png")
plt.close()

# Graph 5: Average ticket revenue and snack spend by genre
genre.plot(kind="bar", figsize=(8, 5))
plt.title("Average Ticket Revenue and Snack Spend by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graph5_genre_spend.png")
plt.close()

# Graph 6: Ticket quantity bands (pie chart)
band_counts = df["Group_Band"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(band_counts.values, labels=band_counts.index, autopct="%1.1f%%")
plt.title("Share of Bookings by Group Size")
plt.tight_layout()
plt.savefig("graph6_group_bands.png")
plt.close()

# Graph 7: Histogram of ticket quantity (shows the large group outliers)
plt.figure(figsize=(7, 5))
plt.hist(df["Ticket_Quantity"], bins=20, color="crimson", edgecolor="black")
plt.axvline(upper, color="black", linestyle="--", label="IQR upper limit")
plt.title("Distribution of Ticket Quantity")
plt.xlabel("Tickets per Booking")
plt.ylabel("Number of Bookings")
plt.legend()
plt.tight_layout()
plt.savefig("graph7_ticket_histogram.png")
plt.close()

print("7 graphs saved as PNG files.")
