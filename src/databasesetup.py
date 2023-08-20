import sqlite3 as sq
import pandas as pd
import json

class Users():
    def __init__(self):
        self.connection = sq.connect('primary.db')
        self.cursor = self.connection.cursor()
        self.create_table()
        self.json_insert()
    
    def create_table(self):
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS users(
            date TEXT,
            userId TEXT,
            username TEXT,
            productId TEXT,
            rating REAL,
            reviewText TEXT,
            feature_matrix TEXT
        )""")
    

    def json_insert(self):
        
        user_data = json.load(open('./final3.json'))
        # print(type(user_data))
        # print(user_data)
        columns = []
        column = []
        for user in user_data:
            for data_obj in user_data[user]:
                column = list(data_obj.keys())
                for col in column:
                    if col not in columns:
                        columns.append(col)
        print(columns)
        value = []
        values = []
        for user in user_data:
            for data_obj in user_data[user]:
                for i in columns:
                    value.append(str(dict(data_obj).get(i)))
                
                
                value.append("Hi")
                self.cursor.execute(
                    """INSERT OR IGNORE INTO users VALUES(?,?,?,?,?,?,?)""", value)
                values.append(list(value))
                value.clear()
        
        # print(values)
        
    
    def fetch_data_from_users(self):
        self.cursor.execute("""SELECT * FROM users""")
        rows = self.cursor.fetchall()
        print(type(rows))

        return rows

# class Products():
#     def __init__(self):
#         self.connection = sq.connect('primary.db')
#         self.cursor = self.connection.cursor()
#         self.create_table()

#     def create_table(self):
#         self.cursor.execute("""CREATE TABLE IF NOT EXISTS products(
#             product_id TEXT PRIMARY KEY,
#             product_name TEXT,
#             category TEXT,
#             discounted_price TEXT ,
#             actual_price TEXT,
#             discounted_percentage TEXT,
#             rating REAL,
#             rating_count REAL,
#             about_product TEXT,
#             user_id TEXT,
#             user_name TEXT,
#             review_id TEXT,
#             review_title TEXT,
#             review_content TEXT,
#             img_link TEXT,
#             product_link TEXT
#         )""")

#     def insert_val_into_products(self,obj):
#         self.cursor.execute("""INSERT OR IGNORE INTO products VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",obj)
#         self.connection.commit()

#     def fetch_data_from_products(self):
#         self.cursor.execute("""SELECT * FROM products""")
#         rows = self.cursor.fetchall()
#         print(type(rows))

#         return rows

# product database testing

# db = Products()
# amazon_data = pd.read_csv('/content/amazon.csv')
# # convert into sql table
# amazon_data.to_sql('products', db.connection, if_exists='replace', index=False)

# records = db.fetch_data_from_products()
# count = 0
# for row in records:
#     print(row)
#     count +=1
#     if(count > 5):
#         break

# db_users = Users()
# user_records = db_users.fetch_data_from_users()

# count = 0
# for row in user_records:
#     print(row)
#     count += 1
#     if count > 5:
#         break
