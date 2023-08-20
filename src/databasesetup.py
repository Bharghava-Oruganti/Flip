import sqlite3
import pandas as pd

class Products():
    def __init__(self):
        self.connection = sqlite3.connect('primary.db')
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS products(
            product_id TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            discounted_price TEXT ,
            actual_price TEXT,
            discounted_percentage TEXT,
            rating REAL,
            rating_count REAL,
            about_product TEXT,
            user_id TEXT,
            user_name TEXT,
            review_id TEXT,
            review_title TEXT,
            review_content TEXT,
            img_link TEXT,
            product_link TEXT
        )""")

    def insert_val_into_products(self,obj):
        self.cursor.execute("""INSERT OR IGNORE INTO products VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",obj)
        self.connection.commit()

    def fetch_data_from_products(self):
        self.cursor.execute("""SELECT * FROM products""")
        rows = self.cursor.fetchall()
        print(type(rows))

        return rows

# product database testing

db = Products()
amazon_data = pd.read_csv('/content/amazon.csv')
# convert into sql table
amazon_data.to_sql('products', db.connection, if_exists='replace', index=False)

records = db.fetch_data_from_products()
# count = 0
# for row in records:
#     print(row)
#     count +=1
#     if(count > 5):
#         break