# Api File
import json
from databasesetup import Users

class API():
    def __init__(self):
        self.user_data = json.load(open('./final3.json'))
        self.product_data = json.load(open('./productfinal.json'))
        self.db_user = Users()
        # self.db_product = Products()
        pass
    
    
    def get_users_attributes(self):
        # print(type(user_data))
        # print(user_data)
        columns = []
        column = []
        for user in self.user_data:
            for data_obj in self.user_data[user]:
                column = list(data_obj.keys())
                for col in column:
                    if col not in columns:
                        columns.append(col)
        
        return columns

    def store_user_features(self,id,obj):
        self.db_user.cursor.execute(
            """UPDATE users SET feature_matrix = ? WHERE userId = ?""", (obj, id))

        print("Feature Matrix Updated! ")

    def get_user_features(self,id):
        self.db_user.cursor.execute(
            """SELECT feature_matrix FROM users WHERE userId = ?""",(id,)
        )
        rows = self.db_user.cursor.fetchall()
        return rows[0]

    def get_products_attributes(self):
        # print(type(user_data))
        # print(user_data)
        columns = []
        column = []
        for product in self.product_data:
            for data_obj in self.product_data[product]:
                column = list(data_obj.keys())
                for col in column:
                    if col not in columns:
                        columns.append(col)

        return columns


api = API()
obj = [1,2,3]
api.store_user_features('A3NHUQ33CFH3VM',str(obj))
row = api.get_user_features('A3NHUQ33CFH3VM')
print(row)