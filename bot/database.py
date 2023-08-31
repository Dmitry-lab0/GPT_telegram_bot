import sqlite3

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('bot.db')
        self.create_table()

    def create_table(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, username TEXT, first_name TEXT, last_name TEXT, conv_context TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS requests
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT, response TEXT)''')
        self.conn.commit()

    def insert_request(self, message, response):
        c = self.conn.cursor()
        c.execute("INSERT INTO requests (message, response) VALUES (?, ?)", (message, response))
        self.conn.commit()
     
    def insert_conv_context(self, conv_context):
        c = self.conn.cursor()
        c.execute("INSERT INTO users (conv_context) VALUES (?)", (conv_context,))
        self.conn.commit()
    
    def select_conv_context(self, user_id):
        c = self.conn.cursor()
        c.execute("SELECT conv_context FROM users WHERE user_id=?", (user_id,))
        conv_context = c.fetchone()
        return conv_context[0] if conv_context else None  
    
    def delete_conv_context(self, user_id):
        c = self.conn.cursor()
        c.execute("UPDATE users SET conv_context='' WHERE user_id=?", (user_id,))
        self.conn.commit()
    
    def insert_user(self, user_id, username, first_name, last_name):
        c = self.conn.cursor()
        conv_context = ''
        c.execute("INSERT INTO users (user_id, username, first_name, last_name, conv_context) VALUES (?, ?, ?, ?, ?)", (user_id, username, first_name, last_name, conv_context))
        self.conn.commit()
        
    def check_user_exists(self, user_id):
        c = self.conn.cursor()
        c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        return user is not None
    
    def close(self):
        self.conn.close()